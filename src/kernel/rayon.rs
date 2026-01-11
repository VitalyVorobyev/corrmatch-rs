//! Rayon-parallel kernels and search helpers (feature-gated).
//!
//! This module provides row-parallel scan functions for unmasked kernels,
//! enabling parallelization for translation-only matching where angle
//! parallelism is not available.

use crate::candidate::topk::{Peak, TopK};
use crate::kernel::ScanParams;
use crate::template::{SsdTemplatePlan, TemplatePlan};
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;
use rayon::prelude::*;

/// Row-parallel full scan for unmasked ZNCC kernel.
///
/// Parallelizes over y-coordinates (rows), with each thread computing
/// scores for all x positions in its assigned rows.
pub fn zncc_unmasked_scan_full_par(
    image: ImageView<'_, u8>,
    tpl: &TemplatePlan,
    angle_idx: usize,
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

    let var_t = tpl.var_t();
    if var_t <= 1e-8 {
        return Ok(Vec::new());
    }
    let t_prime = tpl.t_prime();
    let n = (tpl_width * tpl_height) as f32;

    // Parallel scan over rows
    let row_results: Vec<Vec<Peak>> = (0..=max_y)
        .into_par_iter()
        .map(|y| {
            let mut row_peaks = Vec::new();
            for x in 0..=max_x {
                let mut dot = 0.0f32;
                let mut sum_i = 0.0f32;
                let mut sum_i2 = 0.0f32;

                for ty in 0..tpl_height {
                    let img_row = image.row(y + ty).expect("row within bounds");
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
                    row_peaks.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
            row_peaks
        })
        .collect();

    // Merge results and select top-k
    let mut topk = TopK::new(params.topk);
    for peaks in row_results {
        for peak in peaks {
            topk.push(peak);
        }
    }

    Ok(topk.into_sorted_desc())
}

/// Row-parallel full scan for unmasked SSD kernel.
pub fn ssd_unmasked_scan_full_par(
    image: ImageView<'_, u8>,
    tpl: &SsdTemplatePlan,
    angle_idx: usize,
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
    let data = tpl.data();

    // Parallel scan over rows
    let row_results: Vec<Vec<Peak>> = (0..=max_y)
        .into_par_iter()
        .map(|y| {
            let mut row_peaks = Vec::new();
            for x in 0..=max_x {
                let mut sse = 0.0f32;

                for ty in 0..tpl_height {
                    let img_row = image.row(y + ty).expect("row within bounds");
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
                    row_peaks.push(Peak {
                        x,
                        y,
                        score,
                        angle_idx,
                    });
                }
            }
            row_peaks
        })
        .collect();

    // Merge results and select top-k
    let mut topk = TopK::new(params.topk);
    for peaks in row_results {
        for peak in peaks {
            topk.push(peak);
        }
    }

    Ok(topk.into_sorted_desc())
}
