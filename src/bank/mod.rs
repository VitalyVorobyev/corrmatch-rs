//! Precomputed template assets for coarse-to-fine search.
//!
//! Compiling template assets once amortizes the cost of building pyramids and
//! rotated variants across multiple match calls. Each cached rotation stores
//! precomputed masked plans (ZNCC and SSD) for fast score evaluation.
//! Rotated templates are cached lazily per level; each angle slot is populated
//! at most once and stored in a `OnceLock` for thread-safe reuse when parallel
//! search is introduced later.

mod angles;

pub use angles::AngleGrid;

use crate::image::pyramid::ImagePyramid;
use crate::image::OwnedImage;
use crate::template::rotate::rotate_u8_bilinear_masked;
use crate::template::{
    MaskedSsdTemplatePlan, MaskedTemplatePlan, SsdTemplatePlan, Template, TemplatePlan,
};
use crate::util::{CorrMatchError, CorrMatchResult};
use std::sync::{Arc, OnceLock};

/// Configuration for compiling template assets with rotation support.
#[derive(Clone, Debug)]
pub struct CompileConfig {
    /// Maximum pyramid levels to build.
    pub max_levels: usize,
    /// Coarse rotation step in degrees at level 0.
    pub coarse_step_deg: f32,
    /// Minimum rotation step in degrees across levels.
    pub min_step_deg: f32,
    /// Fill value used for out-of-bounds rotations.
    pub fill_value: u8,
    /// Precompute all rotations for the coarsest level.
    pub precompute_coarsest: bool,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            max_levels: 6,
            coarse_step_deg: 10.0,
            min_step_deg: 0.5,
            fill_value: 0,
            precompute_coarsest: true,
        }
    }
}

impl CompileConfig {
    /// Validates the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> CorrMatchResult<()> {
        if self.max_levels == 0 {
            return Err(CorrMatchError::InvalidConfig {
                reason: "max_levels must be at least 1",
            });
        }
        if !self.coarse_step_deg.is_finite() || self.coarse_step_deg <= 0.0 {
            return Err(CorrMatchError::InvalidConfig {
                reason: "coarse_step_deg must be a positive finite value",
            });
        }
        if !self.min_step_deg.is_finite() || self.min_step_deg <= 0.0 {
            return Err(CorrMatchError::InvalidConfig {
                reason: "min_step_deg must be a positive finite value",
            });
        }
        if self.min_step_deg > self.coarse_step_deg {
            return Err(CorrMatchError::InvalidConfig {
                reason: "min_step_deg must not exceed coarse_step_deg",
            });
        }
        Ok(())
    }
}

/// Configuration for compiling template assets without rotation support.
#[derive(Clone, Debug)]
pub struct CompileConfigNoRot {
    /// Maximum pyramid levels to build.
    pub max_levels: usize,
}

impl Default for CompileConfigNoRot {
    fn default() -> Self {
        Self { max_levels: 6 }
    }
}

pub(crate) struct RotatedTemplate {
    angle_deg: f32,
    zncc: MaskedTemplatePlan,
    ssd: MaskedSsdTemplatePlan,
}

impl RotatedTemplate {
    pub(crate) fn zncc_plan(&self) -> &MaskedTemplatePlan {
        &self.zncc
    }

    pub(crate) fn ssd_plan(&self) -> &MaskedSsdTemplatePlan {
        &self.ssd
    }
}

struct LevelBank {
    grid: AngleGrid,
    slots: Vec<OnceLock<RotatedTemplate>>,
}

/// Compiled template assets with rotation support.
pub struct CompiledTemplateRot {
    levels: Vec<OwnedImage>,
    banks: Vec<LevelBank>,
    unmasked_zncc: Vec<TemplatePlan>,
    unmasked_ssd: Vec<SsdTemplatePlan>,
    cfg: CompileConfig,
}

impl CompiledTemplateRot {
    /// Compiles template assets for matching with rotation support.
    pub fn compile(tpl: &Template, cfg: CompileConfig) -> CorrMatchResult<Self> {
        let pyramid = ImagePyramid::build_u8(tpl.view(), cfg.max_levels)?;
        let levels = pyramid.into_levels();

        let mut unmasked_zncc = Vec::with_capacity(levels.len());
        let mut unmasked_ssd = Vec::with_capacity(levels.len());
        for level in levels.iter() {
            unmasked_zncc.push(TemplatePlan::from_view(level.view())?);
            unmasked_ssd.push(SsdTemplatePlan::from_view(level.view())?);
        }

        let mut banks = Vec::with_capacity(levels.len());
        let coarsest_idx = levels.len().saturating_sub(1);
        for (level_idx, _level) in levels.iter().enumerate() {
            let shift = coarsest_idx.saturating_sub(level_idx);
            let factor = (1u64.checked_shl(shift as u32).unwrap_or(u64::MAX)) as f32;
            let step = (cfg.coarse_step_deg / factor).max(cfg.min_step_deg);
            let grid = AngleGrid::full(step)?;
            let slots = (0..grid.len()).map(|_| OnceLock::new()).collect();
            banks.push(LevelBank { grid, slots });
        }

        if cfg.precompute_coarsest {
            let coarsest_idx = levels.len().saturating_sub(1);
            let coarsest = levels
                .get(coarsest_idx)
                .ok_or(CorrMatchError::IndexOutOfBounds {
                    index: coarsest_idx,
                    len: levels.len(),
                    context: "level",
                })?;
            if let Some(bank) = banks.get_mut(coarsest_idx) {
                for (idx, angle) in bank.grid.iter().enumerate() {
                    let (rotated_img, mask) =
                        rotate_u8_bilinear_masked(coarsest.view(), angle, cfg.fill_value);
                    let mask: Arc<[u8]> = Arc::from(mask);
                    let zncc_plan = MaskedTemplatePlan::from_rotated_parts(
                        rotated_img.view(),
                        mask.clone(),
                        angle,
                    )?;
                    let ssd_plan =
                        MaskedSsdTemplatePlan::from_rotated_parts(rotated_img.view(), mask, angle)?;
                    let rotated = RotatedTemplate {
                        angle_deg: angle,
                        zncc: zncc_plan,
                        ssd: ssd_plan,
                    };
                    let _ = bank.slots[idx].set(rotated);
                }
            }
        }

        Ok(Self {
            levels,
            banks,
            unmasked_zncc,
            unmasked_ssd,
            cfg,
        })
    }

    /// Returns the number of pyramid levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the width and height for a pyramid level.
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        self.levels
            .get(level)
            .map(|img| (img.width(), img.height()))
    }

    /// Returns the angle grid for a pyramid level.
    pub fn angle_grid(&self, level: usize) -> Option<&AngleGrid> {
        self.banks.get(level).map(|bank| &bank.grid)
    }

    /// Returns an unmasked ZNCC template plan for a given level.
    pub fn unmasked_zncc_plan(&self, level: usize) -> CorrMatchResult<&TemplatePlan> {
        self.unmasked_zncc
            .get(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: self.unmasked_zncc.len(),
                context: "level",
            })
    }

    /// Returns an unmasked SSD template plan for a given level.
    pub fn unmasked_ssd_plan(&self, level: usize) -> CorrMatchResult<&SsdTemplatePlan> {
        self.unmasked_ssd
            .get(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: self.unmasked_ssd.len(),
                context: "level",
            })
    }

    pub(crate) fn rotated(
        &self,
        level: usize,
        angle_idx: usize,
    ) -> CorrMatchResult<&RotatedTemplate> {
        let bank = self
            .banks
            .get(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: self.banks.len(),
                context: "level",
            })?;
        let slot = bank
            .slots
            .get(angle_idx)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: angle_idx,
                len: bank.slots.len(),
                context: "angle_idx",
            })?;
        let level_img = self
            .levels
            .get(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: self.levels.len(),
                context: "level",
            })?;
        let angle = bank.grid.angle_at(angle_idx);
        if let Some(rotated) = slot.get() {
            debug_assert!((rotated.angle_deg - angle).abs() < 1e-6);
            debug_assert_eq!(rotated.zncc.width(), level_img.width());
            debug_assert_eq!(rotated.zncc.height(), level_img.height());
            return Ok(rotated);
        }
        let (rotated_img, mask) =
            rotate_u8_bilinear_masked(level_img.view(), angle, self.cfg.fill_value);
        let mask: Arc<[u8]> = Arc::from(mask);
        let zncc_plan =
            MaskedTemplatePlan::from_rotated_parts(rotated_img.view(), mask.clone(), angle)?;
        let ssd_plan = MaskedSsdTemplatePlan::from_rotated_parts(rotated_img.view(), mask, angle)?;
        let rotated = RotatedTemplate {
            angle_deg: angle,
            zncc: zncc_plan,
            ssd: ssd_plan,
        };
        let _ = slot.set(rotated);
        Ok(slot.get().expect("rotated template should be initialized"))
    }
}

/// Compiled template assets without rotation support.
pub struct CompiledTemplateNoRot {
    levels: Vec<OwnedImage>,
    unmasked_zncc: Vec<TemplatePlan>,
    unmasked_ssd: Vec<SsdTemplatePlan>,
}

impl CompiledTemplateNoRot {
    /// Compiles template assets without rotation support.
    pub fn compile(tpl: &Template, cfg: CompileConfigNoRot) -> CorrMatchResult<Self> {
        let pyramid = ImagePyramid::build_u8(tpl.view(), cfg.max_levels)?;
        let levels = pyramid.into_levels();

        let mut unmasked_zncc = Vec::with_capacity(levels.len());
        let mut unmasked_ssd = Vec::with_capacity(levels.len());
        for level in levels.iter() {
            unmasked_zncc.push(TemplatePlan::from_view(level.view())?);
            unmasked_ssd.push(SsdTemplatePlan::from_view(level.view())?);
        }

        Ok(Self {
            levels,
            unmasked_zncc,
            unmasked_ssd,
        })
    }

    /// Returns the number of pyramid levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the width and height for a pyramid level.
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        self.levels
            .get(level)
            .map(|img| (img.width(), img.height()))
    }

    /// Returns an unmasked ZNCC template plan for a given level.
    pub fn unmasked_zncc_plan(&self, level: usize) -> CorrMatchResult<&TemplatePlan> {
        self.unmasked_zncc
            .get(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: self.unmasked_zncc.len(),
                context: "level",
            })
    }

    /// Returns an unmasked SSD template plan for a given level.
    pub fn unmasked_ssd_plan(&self, level: usize) -> CorrMatchResult<&SsdTemplatePlan> {
        self.unmasked_ssd
            .get(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: self.unmasked_ssd.len(),
                context: "level",
            })
    }
}

/// Compiled template assets for rotated or unrotated matching.
///
/// Use `Template::compile`/`CompiledTemplate::compile_rotated` when rotation
/// search is required, or `CompiledTemplate::compile_unrotated` for the fast
/// translation-only path.
pub enum CompiledTemplate {
    /// Rotation-enabled assets.
    Rotated(CompiledTemplateRot),
    /// Rotation-disabled assets.
    Unrotated(CompiledTemplateNoRot),
}

impl CompiledTemplate {
    /// Compiles rotation-enabled template assets.
    pub fn compile_rotated(tpl: &Template, cfg: CompileConfig) -> CorrMatchResult<Self> {
        Ok(Self::Rotated(CompiledTemplateRot::compile(tpl, cfg)?))
    }

    /// Compiles rotation-disabled template assets.
    pub fn compile_unrotated(tpl: &Template, cfg: CompileConfigNoRot) -> CorrMatchResult<Self> {
        Ok(Self::Unrotated(CompiledTemplateNoRot::compile(tpl, cfg)?))
    }

    /// Compiles rotation-enabled template assets (backwards-compatible default).
    pub fn compile(tpl: &Template, cfg: CompileConfig) -> CorrMatchResult<Self> {
        Self::compile_rotated(tpl, cfg)
    }

    /// Returns the number of pyramid levels.
    pub fn num_levels(&self) -> usize {
        match self {
            Self::Rotated(rot) => rot.num_levels(),
            Self::Unrotated(unrot) => unrot.num_levels(),
        }
    }

    /// Returns the width and height for a pyramid level.
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        match self {
            Self::Rotated(rot) => rot.level_size(level),
            Self::Unrotated(unrot) => unrot.level_size(level),
        }
    }

    /// Returns the angle grid for a pyramid level.
    pub fn angle_grid(&self, level: usize) -> Option<&AngleGrid> {
        match self {
            Self::Rotated(rot) => rot.angle_grid(level),
            Self::Unrotated(_) => None,
        }
    }

    /// Returns an unmasked ZNCC template plan for a given level.
    pub fn unmasked_zncc_plan(&self, level: usize) -> CorrMatchResult<&TemplatePlan> {
        match self {
            Self::Rotated(rot) => rot.unmasked_zncc_plan(level),
            Self::Unrotated(unrot) => unrot.unmasked_zncc_plan(level),
        }
    }

    /// Returns an unmasked SSD template plan for a given level.
    pub fn unmasked_ssd_plan(&self, level: usize) -> CorrMatchResult<&SsdTemplatePlan> {
        match self {
            Self::Rotated(rot) => rot.unmasked_ssd_plan(level),
            Self::Unrotated(unrot) => unrot.unmasked_ssd_plan(level),
        }
    }

    /// Returns the rotated template entry for a given level and angle.
    pub(crate) fn rotated(
        &self,
        level: usize,
        angle_idx: usize,
    ) -> CorrMatchResult<&RotatedTemplate> {
        match self {
            Self::Rotated(rot) => rot.rotated(level, angle_idx),
            Self::Unrotated(_) => Err(CorrMatchError::RotationUnavailable {
                reason: "compiled without rotation support",
            }),
        }
    }

    /// Returns a masked ZNCC template plan for a given level and angle.
    pub fn rotated_zncc_plan(
        &self,
        level: usize,
        angle_idx: usize,
    ) -> CorrMatchResult<&MaskedTemplatePlan> {
        Ok(self.rotated(level, angle_idx)?.zncc_plan())
    }

    /// Returns a masked SSD template plan for a given level and angle.
    pub fn rotated_ssd_plan(
        &self,
        level: usize,
        angle_idx: usize,
    ) -> CorrMatchResult<&MaskedSsdTemplatePlan> {
        Ok(self.rotated(level, angle_idx)?.ssd_plan())
    }
}
