//! Scalar reference kernels for score evaluation.

use crate::candidate::topk::{Peak, TopK};
use crate::image::integral::IntegralImages;
use crate::kernel::{Kernel, ScanParams};
use crate::template::{MaskedSsdTemplatePlan, MaskedTemplatePlan, SsdTemplatePlan, TemplatePlan};
use crate::trace::trace_span;
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

/// Scalar masked ZNCC kernel for rotated templates.
pub struct ZnccMaskedScalar;

/// Scalar unmasked ZNCC kernel for rotation-free matching.
///
/// When the `simd` feature is enabled, this kernel may be unused in favor of
/// the SIMD variant.
#[allow(dead_code)]
pub struct ZnccUnmaskedScalar;

/// Scalar masked SSD kernel for rotated templates.
pub struct SsdMaskedScalar;

/// Scalar unmasked SSD kernel for rotation-free matching.
///
/// When the `simd` feature is enabled, this kernel may be unused in favor of
/// the SIMD variant.
#[allow(dead_code)]
pub struct SsdUnmaskedScalar;

impl ZnccMaskedScalar {
    /// Scores a single position using pre-cached image rows.
    ///
    /// This enables multi-angle batch processing where the same image rows
    /// are reused across multiple angle evaluations at the same (x, y) position.
    ///
    /// # Arguments
    /// * `cached_rows` - Pre-fetched image rows covering [y, y + tpl_height).
    /// * `tpl` - The masked template plan.
    /// * `x` - X position in the image.
    /// * `min_var_i` - Minimum variance threshold.
    pub(crate) fn score_at_cached(
        cached_rows: &[&[u8]],
        tpl: &MaskedTemplatePlan,
        x: usize,
        min_var_i: f32,
    ) -> f32 {
        let sum_w = tpl.sum_w();
        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return f32::NEG_INFINITY;
        }

        let valid_coords = tpl.valid_coords();
        let valid_t_prime = tpl.valid_t_prime();

        let mut dot = 0.0f32;
        let mut sum_i = 0.0f32;
        let mut sum_i2 = 0.0f32;

        for (i, coord) in valid_coords.iter().enumerate() {
            let value = cached_rows[coord.y as usize][x + coord.x as usize] as f32;
            dot += valid_t_prime[i] * value;
            sum_i += value;
            sum_i2 += value * value;
        }

        let var_i = sum_i2 - (sum_i * sum_i) / sum_w;
        if var_i <= min_var_i {
            return f32::NEG_INFINITY;
        }

        let denom = (var_t * var_i).sqrt();
        let score = dot / denom;
        if score.is_finite() {
            score
        } else {
            f32::NEG_INFINITY
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &MaskedTemplatePlan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        mut x1: usize,
        mut y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        let _span = trace_span!(
            "zncc_masked_scan",
            angle_idx = angle_idx,
            tpl_w = tpl_width,
            tpl_h = tpl_height
        )
        .entered();

        if params.topk == 0 {
            return Ok(Vec::new());
        }

        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        if x0 > max_x || y0 > max_y {
            return Ok(Vec::new());
        }
        x1 = x1.min(max_x);
        y1 = y1.min(max_y);
        if x0 > x1 || y0 > y1 {
            return Ok(Vec::new());
        }

        let sum_w = tpl.sum_w();
        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return Ok(Vec::new());
        }

        // Use precomputed valid indices for branch-free iteration.
        // This eliminates ~30-50% branch mispredictions from mask checks.
        let valid_coords = tpl.valid_coords();
        let valid_t_prime = tpl.valid_t_prime();

        let mut topk_buf = TopK::new(params.topk);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut dot = 0.0f32;
                let mut sum_i = 0.0f32;
                let mut sum_i2 = 0.0f32;

                // Iterate only over valid pixels (no mask branch).
                for (i, coord) in valid_coords.iter().enumerate() {
                    let img_row = image
                        .row(y + coord.y as usize)
                        .expect("row within bounds for scan");
                    let value = img_row[x + coord.x as usize] as f32;
                    dot += valid_t_prime[i] * value;
                    sum_i += value;
                    sum_i2 += value * value;
                }

                let var_i = sum_i2 - (sum_i * sum_i) / sum_w;
                if var_i <= params.min_var_i {
                    continue;
                }

                let denom = (var_t * var_i).sqrt();
                let score = dot / denom;
                if score.is_finite() && score >= params.min_score {
                    topk_buf.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
        }

        Ok(topk_buf.into_sorted_desc())
    }
}

impl Kernel for ZnccMaskedScalar {
    type Plan = MaskedTemplatePlan;

    fn score_at(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        x: usize,
        y: usize,
        min_var_i: f32,
    ) -> f32 {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return f32::NEG_INFINITY;
        }
        if x > img_width - tpl_width || y > img_height - tpl_height {
            return f32::NEG_INFINITY;
        }

        let sum_w = tpl.sum_w();
        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return f32::NEG_INFINITY;
        }

        // Use precomputed valid indices for branch-free iteration.
        let valid_coords = tpl.valid_coords();
        let valid_t_prime = tpl.valid_t_prime();

        let mut dot = 0.0f32;
        let mut sum_i = 0.0f32;
        let mut sum_i2 = 0.0f32;

        for (i, coord) in valid_coords.iter().enumerate() {
            let img_row = image
                .row(y + coord.y as usize)
                .expect("row within bounds for score");
            let value = img_row[x + coord.x as usize] as f32;
            dot += valid_t_prime[i] * value;
            sum_i += value;
            sum_i2 += value * value;
        }

        let var_i = sum_i2 - (sum_i * sum_i) / sum_w;
        if var_i <= min_var_i {
            return f32::NEG_INFINITY;
        }

        let denom = (var_t * var_i).sqrt();
        let score = dot / denom;
        if score.is_finite() {
            score
        } else {
            f32::NEG_INFINITY
        }
    }

    fn scan_full(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        Self::scan_range(image, tpl, angle_idx, 0, 0, max_x, max_y, params)
    }

    fn scan_roi(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, tpl, angle_idx, x0, y0, x1, y1, params)
    }
}

impl Kernel for SsdMaskedScalar {
    type Plan = MaskedSsdTemplatePlan;

    fn score_at(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        x: usize,
        y: usize,
        _min_var_i: f32,
    ) -> f32 {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return f32::NEG_INFINITY;
        }
        if x > img_width - tpl_width || y > img_height - tpl_height {
            return f32::NEG_INFINITY;
        }

        // Use precomputed valid coordinates for branch-free iteration.
        let valid_coords = tpl.valid_coords();
        let valid_data = tpl.valid_data();
        let mut sse = 0.0f32;

        for (i, coord) in valid_coords.iter().enumerate() {
            let img_row = image
                .row(y + coord.y as usize)
                .expect("row within bounds for score");
            let value = img_row[x + coord.x as usize] as f32;
            let diff = value - valid_data[i];
            sse += diff * diff;
        }

        if sse.is_finite() {
            -sse
        } else {
            f32::NEG_INFINITY
        }
    }

    fn scan_full(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        Self::scan_range(image, tpl, angle_idx, 0, 0, max_x, max_y, params)
    }

    fn scan_roi(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, tpl, angle_idx, x0, y0, x1, y1, params)
    }
}

impl SsdMaskedScalar {
    /// Scores a single position using pre-cached image rows.
    ///
    /// This enables multi-angle batch processing where the same image rows
    /// are reused across multiple angle evaluations at the same (x, y) position.
    ///
    /// # Arguments
    /// * `cached_rows` - Pre-fetched image rows covering [y, y + tpl_height).
    /// * `tpl` - The masked SSD template plan.
    /// * `x` - X position in the image.
    pub(crate) fn score_at_cached(
        cached_rows: &[&[u8]],
        tpl: &MaskedSsdTemplatePlan,
        x: usize,
    ) -> f32 {
        let valid_coords = tpl.valid_coords();
        let valid_data = tpl.valid_data();

        let mut sse = 0.0f32;
        for (i, coord) in valid_coords.iter().enumerate() {
            let value = cached_rows[coord.y as usize][x + coord.x as usize] as f32;
            let diff = value - valid_data[i];
            sse += diff * diff;
        }

        if sse.is_finite() {
            -sse
        } else {
            f32::NEG_INFINITY
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &MaskedSsdTemplatePlan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        mut x1: usize,
        mut y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        let _span = trace_span!(
            "ssd_masked_scan",
            angle_idx = angle_idx,
            tpl_w = tpl_width,
            tpl_h = tpl_height
        )
        .entered();

        if params.topk == 0 {
            return Ok(Vec::new());
        }

        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        if x0 > max_x || y0 > max_y {
            return Ok(Vec::new());
        }
        x1 = x1.min(max_x);
        y1 = y1.min(max_y);
        if x0 > x1 || y0 > y1 {
            return Ok(Vec::new());
        }

        // Use precomputed valid indices for branch-free iteration.
        let valid_coords = tpl.valid_coords();
        let valid_data = tpl.valid_data();
        let mut topk_buf = TopK::new(params.topk);

        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut sse = 0.0f32;

                // Iterate only over valid pixels (no mask branch).
                for (i, coord) in valid_coords.iter().enumerate() {
                    let img_row = image
                        .row(y + coord.y as usize)
                        .expect("row within bounds for scan");
                    let value = img_row[x + coord.x as usize] as f32;
                    let diff = value - valid_data[i];
                    sse += diff * diff;
                }

                let score = -sse;
                if score.is_finite() && score >= params.min_score {
                    topk_buf.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
        }

        Ok(topk_buf.into_sorted_desc())
    }
}

impl ZnccUnmaskedScalar {
    #[allow(dead_code, clippy::too_many_arguments)]
    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        mut x1: usize,
        mut y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        if params.topk == 0 {
            return Ok(Vec::new());
        }

        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        if x0 > max_x || y0 > max_y {
            return Ok(Vec::new());
        }
        x1 = x1.min(max_x);
        y1 = y1.min(max_y);
        if x0 > x1 || y0 > y1 {
            return Ok(Vec::new());
        }

        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return Ok(Vec::new());
        }
        let t_prime = tpl.t_prime();
        let n = (tpl_width * tpl_height) as f32;

        let mut topk_buf = TopK::new(params.topk);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut dot = 0.0f32;
                let mut sum_i = 0.0f32;
                let mut sum_i2 = 0.0f32;

                for ty in 0..tpl_height {
                    let img_row = image.row(y + ty).expect("row within bounds for scan");
                    let base = ty * tpl_width;
                    for tx in 0..tpl_width {
                        let idx = base + tx;
                        let value = img_row[x + tx] as f32;
                        dot += t_prime[idx] * value;
                        sum_i += value;
                        sum_i2 += value * value;
                    }
                }

                let var_i = sum_i2 - (sum_i * sum_i) / n;
                if var_i <= params.min_var_i {
                    continue;
                }

                let denom = (var_t * var_i).sqrt();
                let score = dot / denom;
                if score.is_finite() && score >= params.min_score {
                    topk_buf.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
        }

        Ok(topk_buf.into_sorted_desc())
    }

    #[cfg_attr(feature = "simd", allow(dead_code))]
    #[inline]
    fn dot_at(
        image: ImageView<'_, u8>,
        t_prime: &[f32],
        tpl_width: usize,
        tpl_height: usize,
        x: usize,
        y: usize,
    ) -> f32 {
        let mut dot = 0.0f32;
        for ty in 0..tpl_height {
            let img_row = image.row(y + ty).expect("row within bounds for scan");
            let base = ty * tpl_width;
            for tx in 0..tpl_width {
                let idx = base + tx;
                let value = img_row[x + tx] as f32;
                dot += t_prime[idx] * value;
            }
        }
        dot
    }

    #[cfg_attr(feature = "simd", allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    fn scan_range_integral(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        mut x1: usize,
        mut y1: usize,
        params: ScanParams,
        integrals: &IntegralImages,
    ) -> CorrMatchResult<Vec<Peak>> {
        if params.topk == 0 {
            return Ok(Vec::new());
        }

        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        debug_assert_eq!(integrals.width(), img_width);
        debug_assert_eq!(integrals.height(), img_height);

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        if x0 > max_x || y0 > max_y {
            return Ok(Vec::new());
        }
        x1 = x1.min(max_x);
        y1 = y1.min(max_y);
        if x0 > x1 || y0 > y1 {
            return Ok(Vec::new());
        }

        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return Ok(Vec::new());
        }
        let t_prime = tpl.t_prime();
        let n = (tpl_width * tpl_height) as f32;

        let mut topk_buf = TopK::new(params.topk);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let sum_i = integrals.sum_rect(x, y, tpl_width, tpl_height);
                let sum_i2 = integrals.sumsq_rect(x, y, tpl_width, tpl_height);
                let var_i = sum_i2 - (sum_i * sum_i) / n;
                if var_i <= params.min_var_i {
                    continue;
                }

                let dot = Self::dot_at(image, t_prime, tpl_width, tpl_height, x, y);
                let denom = (var_t * var_i).sqrt();
                let score = dot / denom;
                if score.is_finite() && score >= params.min_score {
                    topk_buf.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
        }

        Ok(topk_buf.into_sorted_desc())
    }

    /// Scans the full valid placement range using integral-image variance pruning.
    #[cfg_attr(feature = "simd", allow(dead_code))]
    pub(crate) fn scan_full_integral(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        params: ScanParams,
        integrals: &IntegralImages,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range_integral(
            image,
            tpl,
            angle_idx,
            0,
            0,
            usize::MAX,
            usize::MAX,
            params,
            integrals,
        )
    }

    /// Scans an ROI using integral-image variance pruning.
    #[cfg_attr(feature = "simd", allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_roi_integral(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
        integrals: &IntegralImages,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range_integral(image, tpl, angle_idx, x0, y0, x1, y1, params, integrals)
    }
}

impl Kernel for ZnccUnmaskedScalar {
    type Plan = TemplatePlan;

    fn score_at(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        x: usize,
        y: usize,
        min_var_i: f32,
    ) -> f32 {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return f32::NEG_INFINITY;
        }
        if x > img_width - tpl_width || y > img_height - tpl_height {
            return f32::NEG_INFINITY;
        }

        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return f32::NEG_INFINITY;
        }
        let t_prime = tpl.t_prime();
        let n = (tpl_width * tpl_height) as f32;

        let mut dot = 0.0f32;
        let mut sum_i = 0.0f32;
        let mut sum_i2 = 0.0f32;

        for ty in 0..tpl_height {
            let img_row = image.row(y + ty).expect("row within bounds for score");
            let base = ty * tpl_width;
            for tx in 0..tpl_width {
                let idx = base + tx;
                let value = img_row[x + tx] as f32;
                dot += t_prime[idx] * value;
                sum_i += value;
                sum_i2 += value * value;
            }
        }

        let var_i = sum_i2 - (sum_i * sum_i) / n;
        if var_i <= min_var_i {
            return f32::NEG_INFINITY;
        }

        let denom = (var_t * var_i).sqrt();
        let score = dot / denom;
        if score.is_finite() {
            score
        } else {
            f32::NEG_INFINITY
        }
    }

    fn scan_full(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        Self::scan_range(image, tpl, angle_idx, 0, 0, max_x, max_y, params)
    }

    fn scan_roi(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, tpl, angle_idx, x0, y0, x1, y1, params)
    }
}

impl Kernel for SsdUnmaskedScalar {
    type Plan = SsdTemplatePlan;

    fn score_at(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        x: usize,
        y: usize,
        _min_var_i: f32,
    ) -> f32 {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return f32::NEG_INFINITY;
        }
        if x > img_width - tpl_width || y > img_height - tpl_height {
            return f32::NEG_INFINITY;
        }

        let data = tpl.data();
        let mut sse = 0.0f32;
        for ty in 0..tpl_height {
            let img_row = image.row(y + ty).expect("row within bounds for score");
            let base = ty * tpl_width;
            for tx in 0..tpl_width {
                let idx = base + tx;
                let value = img_row[x + tx] as f32;
                let diff = value - data[idx];
                sse += diff * diff;
            }
        }

        if sse.is_finite() {
            -sse
        } else {
            f32::NEG_INFINITY
        }
    }

    fn scan_full(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        Self::scan_range(image, tpl, angle_idx, 0, 0, max_x, max_y, params)
    }

    fn scan_roi(
        image: ImageView<'_, u8>,
        tpl: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, tpl, angle_idx, x0, y0, x1, y1, params)
    }
}

impl SsdUnmaskedScalar {
    #[allow(dead_code, clippy::too_many_arguments)]
    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &SsdTemplatePlan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        mut x1: usize,
        mut y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        if params.topk == 0 {
            return Ok(Vec::new());
        }

        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();

        if img_width < tpl_width || img_height < tpl_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x: 0,
                y: 0,
                width: tpl_width,
                height: tpl_height,
                img_width,
                img_height,
            });
        }

        let max_x = img_width - tpl_width;
        let max_y = img_height - tpl_height;
        if x0 > max_x || y0 > max_y {
            return Ok(Vec::new());
        }
        x1 = x1.min(max_x);
        y1 = y1.min(max_y);
        if x0 > x1 || y0 > y1 {
            return Ok(Vec::new());
        }

        let data = tpl.data();
        let mut topk_buf = TopK::new(params.topk);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut sse = 0.0f32;
                for ty in 0..tpl_height {
                    let img_row = image.row(y + ty).expect("row within bounds for scan");
                    let base = ty * tpl_width;
                    for tx in 0..tpl_width {
                        let idx = base + tx;
                        let value = img_row[x + tx] as f32;
                        let diff = value - data[idx];
                        sse += diff * diff;
                    }
                }
                let score = -sse;
                if score.is_finite() && score >= params.min_score {
                    topk_buf.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
        }

        Ok(topk_buf.into_sorted_desc())
    }
}

#[cfg(test)]
mod tests {
    use super::{Kernel, SsdMaskedScalar, SsdUnmaskedScalar, ZnccUnmaskedScalar};
    use crate::image::integral::IntegralImages;
    use crate::kernel::ScanParams;
    use crate::template::{MaskedSsdTemplatePlan, SsdTemplatePlan, TemplatePlan};
    use crate::ImageView;
    use std::collections::HashMap;

    #[test]
    fn unmasked_zncc_scan_matches_bruteforce() {
        let img_width = 6;
        let img_height = 5;
        let mut image = Vec::with_capacity(img_width * img_height);
        for y in 0..img_height {
            for x in 0..img_width {
                image.push(((x * 17 + y * 9 + x * y) & 0xFF) as u8);
            }
        }
        let tpl_width = 3;
        let tpl_height = 2;
        let mut tpl = Vec::with_capacity(tpl_width * tpl_height);
        for y in 0..tpl_height {
            for x in 0..tpl_width {
                tpl.push(((x * 5 + y * 11 + x * y) & 0xFF) as u8);
            }
        }

        let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
        let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
        let plan = TemplatePlan::from_view(tpl_view).unwrap();

        let params = ScanParams {
            topk: 1,
            min_var_i: 1e-8,
            min_score: f32::NEG_INFINITY,
        };
        let best = <ZnccUnmaskedScalar as Kernel>::scan_full(image_view, &plan, 0, params)
            .unwrap()
            .pop()
            .unwrap();

        let t_prime = plan.t_prime();
        let var_t = plan.var_t() as f64;
        let n = (tpl_width * tpl_height) as f64;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_x = 0;
        let mut best_y = 0;
        for y in 0..=(img_height - tpl_height) {
            for x in 0..=(img_width - tpl_width) {
                let mut dot = 0.0f64;
                let mut sum_i = 0.0f64;
                let mut sum_i2 = 0.0f64;
                for ty in 0..tpl_height {
                    let row = image_view.row(y + ty).unwrap();
                    let base = ty * tpl_width;
                    for tx in 0..tpl_width {
                        let idx = base + tx;
                        let value = row[x + tx] as f64;
                        dot += t_prime[idx] as f64 * value;
                        sum_i += value;
                        sum_i2 += value * value;
                    }
                }
                let var_i = sum_i2 - (sum_i * sum_i) / n;
                if var_i <= 1e-8 {
                    continue;
                }
                let score = dot / (var_t * var_i).sqrt();
                if score > best_score {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        assert_eq!(best.x, best_x);
        assert_eq!(best.y, best_y);
        assert!((best.score - best_score as f32).abs() < 1e-5);
    }

    #[test]
    fn unmasked_zncc_integral_scan_matches_scalar() {
        let img_width = 6;
        let img_height = 5;
        let mut image = Vec::with_capacity(img_width * img_height);
        for y in 0..img_height {
            for x in 0..img_width {
                image.push(((x * 17 + y * 9 + x * y + 3) & 0xFF) as u8);
            }
        }
        let tpl_width = 3;
        let tpl_height = 2;
        let mut tpl = Vec::with_capacity(tpl_width * tpl_height);
        for y in 0..tpl_height {
            for x in 0..tpl_width {
                tpl.push(((x * 5 + y * 11 + x * y + 1) & 0xFF) as u8);
            }
        }

        let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
        let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
        let plan = TemplatePlan::from_view(tpl_view).unwrap();
        let placements = (img_width - tpl_width + 1) * (img_height - tpl_height + 1);
        let params = ScanParams {
            topk: placements,
            min_var_i: 1e-8,
            min_score: f32::NEG_INFINITY,
        };

        let scalar =
            <ZnccUnmaskedScalar as Kernel>::scan_full(image_view, &plan, 0, params).unwrap();
        let integrals = IntegralImages::from_u8(image_view).unwrap();
        let integral =
            ZnccUnmaskedScalar::scan_full_integral(image_view, &plan, 0, params, &integrals)
                .unwrap();

        assert_eq!(scalar.len(), integral.len());
        let mut scores = HashMap::with_capacity(scalar.len());
        for peak in scalar {
            scores.insert((peak.x, peak.y), peak.score);
        }
        for peak in integral {
            let score = scores.get(&(peak.x, peak.y)).unwrap();
            assert!((peak.score - score).abs() < 1e-5);
        }
    }

    #[test]
    fn unmasked_ssd_scan_matches_bruteforce() {
        let img_width = 5;
        let img_height = 4;
        let mut image = Vec::with_capacity(img_width * img_height);
        for y in 0..img_height {
            for x in 0..img_width {
                image.push(((x * 9 + y * 5 + x * y) & 0xFF) as u8);
            }
        }
        let tpl_width = 3;
        let tpl_height = 2;
        let mut tpl = Vec::with_capacity(tpl_width * tpl_height);
        for y in 0..tpl_height {
            for x in 0..tpl_width {
                tpl.push(((x * 7 + y * 3 + x * y) & 0xFF) as u8);
            }
        }

        let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
        let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
        let plan = SsdTemplatePlan::from_view(tpl_view).unwrap();

        let params = ScanParams {
            topk: 1,
            min_var_i: 0.0,
            min_score: f32::NEG_INFINITY,
        };
        let best = <SsdUnmaskedScalar as Kernel>::scan_full(image_view, &plan, 0, params)
            .unwrap()
            .pop()
            .unwrap();

        let data = plan.data();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_x = 0;
        let mut best_y = 0;
        for y in 0..=(img_height - tpl_height) {
            for x in 0..=(img_width - tpl_width) {
                let mut sse = 0.0f64;
                for ty in 0..tpl_height {
                    let row = image_view.row(y + ty).unwrap();
                    let base = ty * tpl_width;
                    for tx in 0..tpl_width {
                        let idx = base + tx;
                        let value = row[x + tx] as f64;
                        let diff = value - data[idx] as f64;
                        sse += diff * diff;
                    }
                }
                let score = -sse;
                if score > best_score {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        assert_eq!(best.x, best_x);
        assert_eq!(best.y, best_y);
        assert!((best.score - best_score as f32).abs() < 1e-5);
    }

    #[test]
    fn masked_ssd_score_at_matches_expected() {
        let tpl_width = 3;
        let tpl_height = 3;
        let tpl = vec![
            1u8, 2, 3, //
            4, 5, 6, //
            7, 8, 9,
        ];
        let mask = vec![
            0u8, 0, 0, //
            0, 1, 0, //
            0, 0, 0,
        ];
        let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
        let plan = MaskedSsdTemplatePlan::from_rotated_u8(tpl_view, mask, 0.0).unwrap();

        let image = vec![
            0u8, 0, 0, 0, //
            0, 10, 0, 0, //
            0, 0, 0, 0, //
            0, 0, 0, 0,
        ];
        let image_view = ImageView::from_slice(&image, 4, 4).unwrap();
        let score = <SsdMaskedScalar as Kernel>::score_at(image_view, &plan, 0, 0, 0.0);

        let diff = 10.0f32 - 5.0f32;
        let expected = -(diff * diff);
        assert!((score - expected).abs() < 1e-6);
    }
}
