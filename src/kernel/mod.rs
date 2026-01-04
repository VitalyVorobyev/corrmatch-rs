//! Correlation kernel implementations.

use crate::candidate::topk::Peak;
use crate::util::CorrMatchResult;
use crate::ImageView;

/// Scan configuration for kernel evaluations.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanParams {
    pub(crate) topk: usize,
    pub(crate) min_var_i: f32,
    pub(crate) min_score: f32,
}

/// Kernel trait for scoring and scan operations.
pub(crate) trait Kernel {
    type Plan;

    fn score_at(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        x: usize,
        y: usize,
        min_var_i: f32,
    ) -> f32;

    fn scan_full(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>>;

    #[allow(clippy::too_many_arguments)]
    fn scan_roi(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>>;
}

pub(crate) mod scalar;

#[cfg(feature = "simd")]
pub(crate) mod simd;

#[cfg(feature = "rayon")]
pub(crate) mod rayon;
