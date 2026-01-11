//! SIMD-accelerated kernels using the `wide` crate.
//!
//! This module provides SIMD-optimized kernel implementations for unmasked
//! ZNCC and SSD metrics. The inner template pixel loop is vectorized to
//! process 8 pixels at a time using `f32x8`.

use crate::candidate::topk::{Peak, TopK};
use crate::kernel::{Kernel, ScanParams};
use crate::template::{SsdTemplatePlan, TemplatePlan};
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;
use wide::f32x8;

const LANES: usize = 8;

/// Load 8 u8 values and convert to f32x8.
#[inline]
fn load_u8x8_as_f32x8(slice: &[u8]) -> f32x8 {
    f32x8::from([
        slice[0] as f32,
        slice[1] as f32,
        slice[2] as f32,
        slice[3] as f32,
        slice[4] as f32,
        slice[5] as f32,
        slice[6] as f32,
        slice[7] as f32,
    ])
}

/// Load 8 f32 values into f32x8.
#[inline]
fn load_f32x8(slice: &[f32]) -> f32x8 {
    f32x8::from([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ])
}

/// Horizontal sum of f32x8.
#[inline]
fn hsum(v: f32x8) -> f32 {
    let arr = v.to_array();
    arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
}

/// SIMD-accelerated unmasked ZNCC kernel.
pub struct ZnccUnmaskedSimd;

impl ZnccUnmaskedSimd {
    /// Computes ZNCC score at a single position using SIMD.
    fn score_at_simd(
        image: ImageView<'_, u8>,
        tpl: &TemplatePlan,
        x: usize,
        y: usize,
        min_var_i: f32,
    ) -> f32 {
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        let t_prime = tpl.t_prime();
        let var_t = tpl.var_t();
        let n = (tpl_width * tpl_height) as f32;

        let mut dot_vec = f32x8::ZERO;
        let mut sum_i_vec = f32x8::ZERO;
        let mut sum_i2_vec = f32x8::ZERO;

        let mut dot_s = 0.0f32;
        let mut sum_i_s = 0.0f32;
        let mut sum_i2_s = 0.0f32;

        let simd_end = tpl_width / LANES * LANES;

        for ty in 0..tpl_height {
            let img_row = match image.row(y + ty) {
                Some(row) => row,
                None => return f32::NEG_INFINITY,
            };
            let base = ty * tpl_width;

            // SIMD portion: process 8 pixels at a time
            let mut tx = 0;
            while tx < simd_end {
                let img_vals = load_u8x8_as_f32x8(&img_row[x + tx..]);
                let tpl_vals = load_f32x8(&t_prime[base + tx..]);

                dot_vec += tpl_vals * img_vals;
                sum_i_vec += img_vals;
                sum_i2_vec += img_vals * img_vals;

                tx += LANES;
            }

            // Scalar remainder
            while tx < tpl_width {
                let value = img_row[x + tx] as f32;
                let idx = base + tx;
                dot_s += t_prime[idx] * value;
                sum_i_s += value;
                sum_i2_s += value * value;
                tx += 1;
            }
        }

        // Horizontal reduction + scalar remainder
        let dot = hsum(dot_vec) + dot_s;
        let sum_i = hsum(sum_i_vec) + sum_i_s;
        let sum_i2 = hsum(sum_i2_vec) + sum_i2_s;

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

    #[allow(clippy::too_many_arguments)]
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

        let mut topk_buf = TopK::new(params.topk);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let score = Self::score_at_simd(image, tpl, x, y, params.min_var_i);
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

impl Kernel for ZnccUnmaskedSimd {
    type Plan = TemplatePlan;

    fn score_at(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        x: usize,
        y: usize,
        min_var_i: f32,
    ) -> f32 {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = plan.width();
        let tpl_height = plan.height();

        if img_width < tpl_width || img_height < tpl_height {
            return f32::NEG_INFINITY;
        }
        if x > img_width - tpl_width || y > img_height - tpl_height {
            return f32::NEG_INFINITY;
        }

        Self::score_at_simd(image, plan, x, y, min_var_i)
    }

    fn scan_full(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, plan, angle_idx, 0, 0, usize::MAX, usize::MAX, params)
    }

    fn scan_roi(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, plan, angle_idx, x0, y0, x1, y1, params)
    }
}

/// SIMD-accelerated unmasked SSD kernel.
pub struct SsdUnmaskedSimd;

impl SsdUnmaskedSimd {
    /// Computes SSD score at a single position using SIMD.
    fn score_at_simd(image: ImageView<'_, u8>, tpl: &SsdTemplatePlan, x: usize, y: usize) -> f32 {
        let tpl_width = tpl.width();
        let tpl_height = tpl.height();
        let data = tpl.data();

        let mut sse_vec = f32x8::ZERO;
        let mut sse_s = 0.0f32;

        let simd_end = tpl_width / LANES * LANES;

        for ty in 0..tpl_height {
            let img_row = match image.row(y + ty) {
                Some(row) => row,
                None => return f32::NEG_INFINITY,
            };
            let base = ty * tpl_width;

            // SIMD portion
            let mut tx = 0;
            while tx < simd_end {
                let img_vals = load_u8x8_as_f32x8(&img_row[x + tx..]);
                let tpl_vals = load_f32x8(&data[base + tx..]);
                let diff = img_vals - tpl_vals;
                sse_vec += diff * diff;
                tx += LANES;
            }

            // Scalar remainder
            while tx < tpl_width {
                let value = img_row[x + tx] as f32;
                let diff = value - data[base + tx];
                sse_s += diff * diff;
                tx += 1;
            }
        }

        let sse = hsum(sse_vec) + sse_s;
        if sse.is_finite() {
            -sse
        } else {
            f32::NEG_INFINITY
        }
    }

    #[allow(clippy::too_many_arguments)]
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

        let mut topk_buf = TopK::new(params.topk);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let score = Self::score_at_simd(image, tpl, x, y);
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

impl Kernel for SsdUnmaskedSimd {
    type Plan = SsdTemplatePlan;

    fn score_at(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        x: usize,
        y: usize,
        _min_var_i: f32,
    ) -> f32 {
        let img_width = image.width();
        let img_height = image.height();
        let tpl_width = plan.width();
        let tpl_height = plan.height();

        if img_width < tpl_width || img_height < tpl_height {
            return f32::NEG_INFINITY;
        }
        if x > img_width - tpl_width || y > img_height - tpl_height {
            return f32::NEG_INFINITY;
        }

        Self::score_at_simd(image, plan, x, y)
    }

    fn scan_full(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, plan, angle_idx, 0, 0, usize::MAX, usize::MAX, params)
    }

    fn scan_roi(
        image: ImageView<'_, u8>,
        plan: &Self::Plan,
        angle_idx: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        params: ScanParams,
    ) -> CorrMatchResult<Vec<Peak>> {
        Self::scan_range(image, plan, angle_idx, x0, y0, x1, y1, params)
    }
}
