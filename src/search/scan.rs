//! Dense scan over search regions.

use crate::candidate::topk::{Peak, TopK};
use crate::template::MaskedTemplatePlan;
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

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
    if topk == 0 {
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

    let sum_w = tpl.sum_w();
    let var_t = tpl.var_t();
    let t_prime = tpl.t_prime();
    let mask = tpl.mask();

    let mut topk_buf = TopK::new(topk);
    let max_y = img_height - tpl_height;
    let max_x = img_width - tpl_width;
    for y in 0..=max_y {
        for x in 0..=max_x {
            let mut dot = 0.0f32;
            let mut sum_i = 0.0f32;
            let mut sum_i2 = 0.0f32;

            for ty in 0..tpl_height {
                let img_row = image.row(y + ty).expect("row within bounds for scan");
                let base = ty * tpl_width;
                for tx in 0..tpl_width {
                    let idx = base + tx;
                    let w = if mask[idx] == 0 { 0.0 } else { 1.0 };
                    let value = img_row[x + tx] as f32;
                    dot += t_prime[idx] * value;
                    sum_i += w * value;
                    sum_i2 += w * value * value;
                }
            }

            let var_i = sum_i2 - (sum_i * sum_i) / sum_w;
            if var_i <= 1e-8 {
                continue;
            }

            let denom = (var_t * var_i).sqrt();
            let score = dot / denom;
            if score.is_finite() {
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
