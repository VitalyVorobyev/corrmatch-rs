//! Precomputed template assets for coarse-to-fine search.
//!
//! Compiling template assets once amortizes the cost of building pyramids and
//! rotated variants across multiple match calls. Each cached rotation stores a
//! precomputed masked plan (`t'`, `sum_w`, `var_t`) for fast ZNCC evaluation.
//! Rotated templates are cached lazily per level; each angle slot is populated
//! at most once and stored in a `OnceLock` for thread-safe reuse when parallel
//! search is introduced later.

mod angles;

pub use angles::AngleGrid;

use crate::image::pyramid::ImagePyramid;
use crate::image::OwnedImage;
use crate::template::rotate::rotate_u8_bilinear_masked;
use crate::template::{MaskedTemplatePlan, Template};
use crate::util::{CorrMatchError, CorrMatchResult};
use std::sync::OnceLock;

/// Configuration for compiling template assets.
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

pub(crate) struct RotatedTemplate {
    angle_deg: f32,
    plan: MaskedTemplatePlan,
}

impl RotatedTemplate {
    pub(crate) fn plan(&self) -> &MaskedTemplatePlan {
        &self.plan
    }
}

struct LevelBank {
    grid: AngleGrid,
    slots: Vec<OnceLock<RotatedTemplate>>,
}

/// Compiled template assets with a pyramid and per-level rotation grids.
pub struct CompiledTemplate {
    levels: Vec<OwnedImage>,
    banks: Vec<LevelBank>,
    cfg: CompileConfig,
}

impl CompiledTemplate {
    /// Compiles template assets for matching.
    pub fn compile(tpl: &Template, cfg: CompileConfig) -> CorrMatchResult<Self> {
        let pyramid = ImagePyramid::build_u8(tpl.view(), cfg.max_levels)?;
        let levels = pyramid.into_levels();

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
                    let plan =
                        MaskedTemplatePlan::from_rotated_u8(rotated_img.view(), mask, angle)?;
                    let rotated = RotatedTemplate {
                        angle_deg: angle,
                        plan,
                    };
                    let _ = bank.slots[idx].set(rotated);
                }
            }
        }

        Ok(Self { levels, banks, cfg })
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

    /// Returns a masked template plan for a given level and angle.
    pub fn rotated_plan(
        &self,
        level: usize,
        angle_idx: usize,
    ) -> CorrMatchResult<&MaskedTemplatePlan> {
        Ok(&self.rotated(level, angle_idx)?.plan)
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
            debug_assert_eq!(rotated.plan.width(), level_img.width());
            debug_assert_eq!(rotated.plan.height(), level_img.height());
            return Ok(rotated);
        }
        let (rotated_img, mask) =
            rotate_u8_bilinear_masked(level_img.view(), angle, self.cfg.fill_value);
        let plan = MaskedTemplatePlan::from_rotated_u8(rotated_img.view(), mask, angle)?;
        let rotated = RotatedTemplate {
            angle_deg: angle,
            plan,
        };
        let _ = slot.set(rotated);
        Ok(slot.get().expect("rotated template should be initialized"))
    }
}
