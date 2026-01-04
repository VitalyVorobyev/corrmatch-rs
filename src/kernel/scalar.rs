//! Scalar reference kernels for score evaluation.

use crate::candidate::topk::{Peak, TopK};
use crate::kernel::{Kernel, ScanParams};
use crate::template::{MaskedTemplatePlan, TemplatePlan};
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

/// Scalar masked ZNCC kernel for rotated templates.
pub(crate) struct ZnccMaskedScalar;

/// Scalar unmasked ZNCC kernel for rotation-free matching.
pub(crate) struct ZnccUnmaskedScalar;

impl ZnccMaskedScalar {
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

        let sum_w = tpl.sum_w();
        let var_t = tpl.var_t();
        if var_t <= 1e-8 {
            return Ok(Vec::new());
        }
        let t_prime = tpl.t_prime();
        let mask = tpl.mask();

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
                        if mask[idx] == 0 {
                            continue;
                        }
                        let value = img_row[x + tx] as f32;
                        dot += t_prime[idx] * value;
                        sum_i += value;
                        sum_i2 += value * value;
                    }
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
        let t_prime = tpl.t_prime();
        let mask = tpl.mask();

        let mut dot = 0.0f32;
        let mut sum_i = 0.0f32;
        let mut sum_i2 = 0.0f32;

        for ty in 0..tpl_height {
            let img_row = image.row(y + ty).expect("row within bounds for score");
            let base = ty * tpl_width;
            for tx in 0..tpl_width {
                let idx = base + tx;
                if mask[idx] == 0 {
                    continue;
                }
                let value = img_row[x + tx] as f32;
                dot += t_prime[idx] * value;
                sum_i += value;
                sum_i2 += value * value;
            }
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

impl ZnccUnmaskedScalar {
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

#[cfg(test)]
mod tests {
    use super::{Kernel, ZnccUnmaskedScalar};
    use crate::kernel::ScanParams;
    use crate::template::TemplatePlan;
    use crate::ImageView;

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
}
