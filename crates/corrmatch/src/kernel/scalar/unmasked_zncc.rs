use crate::candidate::topk::{Peak, TopK};
use crate::image::integral::IntegralImages;
use crate::kernel::{Kernel, ScanParams, ScanRoi};
use crate::template::TemplatePlan;
use crate::util::CorrMatchResult;
use crate::ImageView;

use super::common::clamp_scan_roi;

/// Scalar unmasked ZNCC kernel for rotation-free matching.
pub struct ZnccUnmaskedScalar;

impl ZnccUnmaskedScalar {
    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        roi: ScanRoi,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        if params.topk == 0 {
            return Ok(Vec::new());
        }

        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        let roi = match clamp_scan_roi(image, tpl_width, tpl_height, roi)? {
            Some(roi) => roi,
            None => return Ok(Vec::new()),
        };

        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return Ok(Vec::new());
        }
        let t_prime = tpl.zero_mean();
        let n = (tpl_width * tpl_height) as f32;

        let mut topk_buf = TopK::new(params.topk);
        for y in roi.y0..=roi.y1 {
            for x in roi.x0..=roi.x1 {
                let mut dot = 0.0f32;
                let mut sum_i = 0.0f32;
                let mut sum_i2 = 0.0f32;

                for ty in 0..tpl_height {
                    // Invariant: roi.y1 <= img_height - tpl_height (enforced by
                    // clamp_scan_roi), so y + ty < img_height for all ty < tpl_height.
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
            // Invariant: caller passes y <= img_height - tpl_height, so y + ty < img_height.
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

    fn scan_range_integral(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        roi: ScanRoi,
        params: ScanParams,
        integrals: &IntegralImages,
    ) -> CorrMatchResult<Vec<Peak>> {
        if params.topk == 0 {
            return Ok(Vec::new());
        }

        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        let roi = match clamp_scan_roi(image, tpl_width, tpl_height, roi)? {
            Some(roi) => roi,
            None => return Ok(Vec::new()),
        };

        debug_assert_eq!(integrals.width(), image.width());
        debug_assert_eq!(integrals.height(), image.height());

        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return Ok(Vec::new());
        }
        let t_prime = tpl.zero_mean();
        let n = (tpl_width * tpl_height) as f32;

        let mut topk_buf = TopK::new(params.topk);
        for y in roi.y0..=roi.y1 {
            for x in roi.x0..=roi.x1 {
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
            ScanRoi::new(0, 0, usize::MAX, usize::MAX),
            params,
            integrals,
        )
    }

    /// Scans an ROI using integral-image variance pruning.
    pub(crate) fn scan_roi_integral(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        angle_idx: usize,
        roi: ScanRoi,
        params: ScanParams,
        integrals: &IntegralImages,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range_integral(image, tpl, angle_idx, roi, params, integrals)
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
        let t_prime = tpl.zero_mean();
        let n = (tpl_width * tpl_height) as f32;

        let mut dot = 0.0f32;
        let mut sum_i = 0.0f32;
        let mut sum_i2 = 0.0f32;

        for ty in 0..tpl_height {
            // Invariant: score_at is called with y <= img_height - tpl_height,
            // so y + ty < img_height for all ty < tpl_height.
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
        Self::scan_range(
            image,
            tpl,
            angle_idx,
            ScanRoi::new(0, 0, usize::MAX, usize::MAX),
            params,
        )
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
        Self::scan_range(image, tpl, angle_idx, ScanRoi::new(x0, y0, x1, y1), params)
    }
}
