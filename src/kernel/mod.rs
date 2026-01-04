//! Correlation kernel implementations.

use crate::candidate::topk::Peak;
use crate::util::CorrMatchResult;
use crate::ImageView;

/// Scan configuration for kernel evaluations.
#[derive(Clone, Copy, Debug)]
pub struct ScanParams {
    /// Maximum number of peaks to retain.
    pub topk: usize,
    /// Minimum variance threshold for the image window (ZNCC only).
    pub min_var_i: f32,
    /// Minimum score threshold (discard below this value).
    pub min_score: f32,
}

/// Kernel trait for scoring and scan operations.
pub trait Kernel {
    type Plan;

    /// Computes the score at a single placement (top-left coordinates).
    fn score_at(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        x: usize,
        y: usize,
        min_var_i: f32,
    ) -> f32;

    /// Scans the full valid placement range and returns top-K peaks.
    fn scan_full(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>>;

    /// Scans an ROI of placement coordinates and returns top-K peaks.
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

pub mod scalar;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "rayon")]
pub mod rayon;
