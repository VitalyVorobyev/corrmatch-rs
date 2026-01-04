//! Dense scan over search regions.

use crate::candidate::topk::Peak;
use crate::kernel::scalar::ZnccMaskedScalar;
use crate::kernel::{Kernel, ScanParams};
use crate::template::MaskedTemplatePlan;
use crate::util::CorrMatchResult;
use crate::ImageView;

/// Computes the masked ZNCC score for a single placement.
///
/// The placement coordinates are top-left offsets into the image. If the
/// placement is invalid or the local variance is too small, returns
/// `f32::NEG_INFINITY`.
pub fn score_masked_zncc_at(
    image: ImageView<'_, u8>,
    tpl: &MaskedTemplatePlan,
    x: usize,
    y: usize,
    min_var_i: f32,
) -> f32 {
    <ZnccMaskedScalar as Kernel>::score_at(image, tpl, x, y, min_var_i)
}

/// Scans an image with a masked ZNCC template and returns the top-K peaks.
///
/// The score is expected to lie in approximately `[-1, 1]` for normalized data.
/// Masked statistics (`sum_w`, `var_t`, `t_prime`) are precomputed in the plan.
pub fn scan_masked_zncc_scalar(
    image: ImageView<'_, u8>,
    tpl: &MaskedTemplatePlan,
    angle_idx: usize,
    topk: usize,
) -> CorrMatchResult<Vec<Peak>> {
    scan_masked_zncc_scalar_full(image, tpl, angle_idx, topk, 1e-8, f32::NEG_INFINITY)
}

/// Scans the full valid placement range for a masked ZNCC template.
pub fn scan_masked_zncc_scalar_full(
    image: ImageView<'_, u8>,
    tpl: &MaskedTemplatePlan,
    angle_idx: usize,
    topk: usize,
    min_var_i: f32,
    min_score: f32,
) -> CorrMatchResult<Vec<Peak>> {
    let params = ScanParams {
        topk,
        min_var_i,
        min_score,
    };
    <ZnccMaskedScalar as Kernel>::scan_full(image, tpl, angle_idx, params)
}

/// Scans an ROI of placement coordinates for a masked ZNCC template.
#[allow(clippy::too_many_arguments)]
pub fn scan_masked_zncc_scalar_roi(
    image: ImageView<'_, u8>,
    tpl: &MaskedTemplatePlan,
    angle_idx: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    topk: usize,
    min_var_i: f32,
    min_score: f32,
) -> CorrMatchResult<Vec<Peak>> {
    let params = ScanParams {
        topk,
        min_var_i,
        min_score,
    };
    <ZnccMaskedScalar as Kernel>::scan_roi(image, tpl, angle_idx, x0, y0, x1, y1, params)
}
