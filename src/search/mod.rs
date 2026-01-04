//! Search strategies for locating template matches.
//!
//! The scan module provides baseline scalar ZNCC evaluation.

mod coarse;
mod refine;
pub(crate) mod scan;

use crate::bank::CompiledTemplate;
use crate::image::pyramid::ImagePyramid;
use crate::search::coarse::{coarse_search_level, coarse_search_level_unmasked};
#[cfg(feature = "rayon")]
use crate::search::coarse::{coarse_search_level_par, coarse_search_level_unmasked_par};
use crate::search::refine::{
    refine_final_match, refine_final_match_unmasked, refine_to_finer_level,
    refine_to_finer_level_unmasked,
};
#[cfg(feature = "rayon")]
use crate::search::refine::{refine_to_finer_level_par, refine_to_finer_level_unmasked_par};
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

/// Matching metric selector (SSD is planned but not implemented).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Zero-mean normalized cross-correlation.
    Zncc,
    /// Sum of squared differences (placeholder).
    Ssd,
}

/// Controls whether rotation is searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationMode {
    /// Skip rotation and use the unmasked fast path.
    Disabled,
    /// Enable rotation search using masked kernels.
    Enabled,
}

/// Configuration for the coarse-to-fine matcher pipeline.
#[derive(Clone, Debug)]
pub struct MatchConfig {
    /// Matching metric to use.
    pub metric: Metric,
    /// Whether rotation search is enabled.
    pub rotation: RotationMode,
    /// Enables parallel search when the `rayon` feature is available.
    ///
    /// When the feature is disabled, this flag is ignored and execution stays sequential.
    pub parallel: bool,
    /// Maximum pyramid levels to build for the image.
    pub max_image_levels: usize,
    /// Beam width kept per level after merge and NMS.
    pub beam_width: usize,
    /// Top-M peaks per angle at the coarsest level.
    pub per_angle_topk: usize,
    /// Spatial NMS radius in pixels for the current level.
    pub nms_radius: usize,
    /// Refinement ROI radius in pixels for the current level.
    pub roi_radius: usize,
    /// Angle neighborhood half-range in multiples of the grid step.
    ///
    /// Ignored when rotation is disabled.
    pub angle_half_range_steps: usize,
    /// Minimum variance for image patches.
    pub min_var_i: f32,
    /// Minimum score threshold (discard below this value).
    pub min_score: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            metric: Metric::Zncc,
            rotation: RotationMode::Disabled,
            parallel: false,
            max_image_levels: 6,
            beam_width: 8,
            per_angle_topk: 3,
            nms_radius: 6,
            roi_radius: 8,
            angle_half_range_steps: 1,
            min_var_i: 1e-8,
            min_score: f32::NEG_INFINITY,
        }
    }
}

impl MatchConfig {
    pub(crate) fn use_parallel(&self) -> bool {
        self.parallel && cfg!(feature = "rayon")
    }
}

/// Match result for the finest pyramid level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    /// Refined top-left x coordinate of the template placement (level 0).
    pub x: f32,
    /// Refined top-left y coordinate of the template placement (level 0).
    pub y: f32,
    /// Estimated rotation angle in degrees.
    pub angle_deg: f32,
    /// Masked ZNCC score for the best match.
    pub score: f32,
}

/// Matcher that runs coarse-to-fine search using a compiled template.
pub struct Matcher {
    compiled: CompiledTemplate,
    cfg: MatchConfig,
}

impl Matcher {
    /// Creates a matcher with default configuration.
    pub fn new(compiled: CompiledTemplate) -> Self {
        Self {
            compiled,
            cfg: MatchConfig::default(),
        }
    }

    /// Replaces the matcher configuration.
    pub fn with_config(mut self, cfg: MatchConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Matches a template against an image and returns the best candidate.
    ///
    /// When rotation is disabled, angle-related settings are ignored.
    pub fn match_image(&self, image: ImageView<'_, u8>) -> CorrMatchResult<Match> {
        if self.cfg.metric != Metric::Zncc {
            return Err(CorrMatchError::UnsupportedMetric { metric: "ssd" });
        }

        let use_parallel = self.cfg.use_parallel();
        let pyramid = ImagePyramid::build_u8(image, self.cfg.max_image_levels)?;
        let num_levels = pyramid.levels().len().min(self.compiled.num_levels());
        if num_levels == 0 {
            return Err(CorrMatchError::InvalidDimensions {
                width: image.width(),
                height: image.height(),
            });
        }

        let coarsest = num_levels - 1;
        let coarse_view = pyramid
            .level(coarsest)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: coarsest,
                len: pyramid.levels().len(),
                context: "image level",
            })?;
        let mut seeds = match self.cfg.rotation {
            RotationMode::Enabled => {
                if use_parallel {
                    #[cfg(feature = "rayon")]
                    {
                        coarse_search_level_par(coarse_view, &self.compiled, coarsest, &self.cfg)?
                    }
                    #[cfg(not(feature = "rayon"))]
                    {
                        coarse_search_level(coarse_view, &self.compiled, coarsest, &self.cfg)?
                    }
                } else {
                    coarse_search_level(coarse_view, &self.compiled, coarsest, &self.cfg)?
                }
            }
            RotationMode::Disabled => {
                if use_parallel {
                    #[cfg(feature = "rayon")]
                    {
                        coarse_search_level_unmasked_par(
                            coarse_view,
                            &self.compiled,
                            coarsest,
                            &self.cfg,
                        )?
                    }
                    #[cfg(not(feature = "rayon"))]
                    {
                        coarse_search_level_unmasked(
                            coarse_view,
                            &self.compiled,
                            coarsest,
                            &self.cfg,
                        )?
                    }
                } else {
                    coarse_search_level_unmasked(coarse_view, &self.compiled, coarsest, &self.cfg)?
                }
            }
        };
        if seeds.is_empty() {
            return Err(CorrMatchError::NoCandidates {
                reason: "no coarse candidates",
            });
        }

        for level in (0..coarsest).rev() {
            let level_view = pyramid
                .level(level)
                .ok_or(CorrMatchError::IndexOutOfBounds {
                    index: level,
                    len: pyramid.levels().len(),
                    context: "image level",
                })?;
            seeds = match self.cfg.rotation {
                RotationMode::Enabled => {
                    if use_parallel {
                        #[cfg(feature = "rayon")]
                        {
                            refine_to_finer_level_par(
                                level_view,
                                &self.compiled,
                                level,
                                &seeds,
                                &self.cfg,
                            )?
                        }
                        #[cfg(not(feature = "rayon"))]
                        {
                            refine_to_finer_level(
                                level_view,
                                &self.compiled,
                                level,
                                &seeds,
                                &self.cfg,
                            )?
                        }
                    } else {
                        refine_to_finer_level(level_view, &self.compiled, level, &seeds, &self.cfg)?
                    }
                }
                RotationMode::Disabled => {
                    if use_parallel {
                        #[cfg(feature = "rayon")]
                        {
                            refine_to_finer_level_unmasked_par(
                                level_view,
                                &self.compiled,
                                level,
                                &seeds,
                                &self.cfg,
                            )?
                        }
                        #[cfg(not(feature = "rayon"))]
                        {
                            refine_to_finer_level_unmasked(
                                level_view,
                                &self.compiled,
                                level,
                                &seeds,
                                &self.cfg,
                            )?
                        }
                    } else {
                        refine_to_finer_level_unmasked(
                            level_view,
                            &self.compiled,
                            level,
                            &seeds,
                            &self.cfg,
                        )?
                    }
                }
            };
            if seeds.is_empty() {
                return Err(CorrMatchError::NoCandidates {
                    reason: "no candidates after refinement",
                });
            }
        }

        let best = seeds[0];
        let refined = match self.cfg.rotation {
            RotationMode::Enabled => refine_final_match(image, &self.compiled, 0, best, &self.cfg),
            RotationMode::Disabled => {
                refine_final_match_unmasked(image, &self.compiled, 0, best, &self.cfg)
            }
        };
        Ok(refined.unwrap_or(Match {
            x: best.x as f32,
            y: best.y as f32,
            angle_deg: best.angle_deg,
            score: best.score,
        }))
    }
}
