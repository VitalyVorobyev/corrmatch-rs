use crate::candidate::topk::{Peak, TopK};
use crate::kernel::{Kernel, ScanParams, ScanRoi};
use crate::template::SsdTemplatePlan;
use crate::util::CorrMatchResult;
use crate::ImageView;

use super::common::clamp_scan_roi;

/// Scalar unmasked SSD kernel for rotation-free matching.
pub struct SsdUnmaskedScalar;

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

impl SsdUnmaskedScalar {
    fn scan_range(
        image: ImageView<'_, u8>,
        tpl: &SsdTemplatePlan,
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

        let data = tpl.data();
        let mut topk_buf = TopK::new(params.topk);
        for y in roi.y0..=roi.y1 {
            for x in roi.x0..=roi.x1 {
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
