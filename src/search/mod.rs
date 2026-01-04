//! Search strategies for locating template matches.
//!
//! The scan module provides baseline scalar ZNCC evaluation.

mod coarse;
mod refine;
pub(crate) mod scan;

use crate::bank::CompiledTemplate;
use crate::image::pyramid::ImagePyramid;
use crate::search::coarse::coarse_search_level;
use crate::search::refine::{refine_final_match, refine_to_finer_level};
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

/// Configuration for the coarse-to-fine matcher pipeline.
#[derive(Clone, Debug)]
pub struct MatchConfig {
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
    pub angle_half_range_steps: usize,
    /// Minimum variance for image patches.
    pub min_var_i: f32,
    /// Minimum score threshold (discard below this value).
    pub min_score: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
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
    pub fn match_image(&self, image: ImageView<'_, u8>) -> CorrMatchResult<Match> {
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
        let mut seeds = coarse_search_level(coarse_view, &self.compiled, coarsest, &self.cfg)?;
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
            seeds = refine_to_finer_level(level_view, &self.compiled, level, &seeds, &self.cfg)?;
            if seeds.is_empty() {
                return Err(CorrMatchError::NoCandidates {
                    reason: "no candidates after refinement",
                });
            }
        }

        let best = seeds[0];
        let refined = refine_final_match(image, &self.compiled, 0, best, &self.cfg);
        Ok(refined.unwrap_or(Match {
            x: best.x as f32,
            y: best.y as f32,
            angle_deg: best.angle_deg,
            score: best.score,
        }))
    }
}
