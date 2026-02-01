use crate::candidate::topk::{Peak, TopK};
use crate::kernel::{Kernel, ScanParams, ScanRoi};
use crate::template::MaskedTemplatePlan;
use crate::util::CorrMatchResult;
use crate::ImageView;

use super::common::clamp_scan_roi;

/// Scalar masked ZNCC kernel for rotated templates.
pub struct ZnccMaskedScalar;

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

    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &MaskedTemplatePlan,
        angle_idx: usize,
        roi: ScanRoi,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
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

        let roi = match clamp_scan_roi(image, tpl_width, tpl_height, roi)? {
            Some(roi) => roi,
            None => return Ok(Vec::new()),
        };

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
        for y in roi.y0..=roi.y1 {
            for x in roi.x0..=roi.x1 {
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
