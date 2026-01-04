//! Refinement search around coarse candidates.
//!
//! Refinement upsamples coarse candidates to finer levels and searches a local
//! ROI and angle neighborhood to improve position and angle estimates.

use crate::bank::CompiledTemplate;
use crate::candidate::topk::Peak;
use crate::search::scan::scan_masked_zncc_scalar_roi;
use crate::search::MatchConfig;
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub(crate) level: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) angle_idx: usize,
    pub(crate) angle_deg: f32,
    pub(crate) score: f32,
}

impl Candidate {
    pub(crate) fn to_peak(self) -> Peak {
        Peak {
            x: self.x,
            y: self.y,
            score: self.score,
            angle_idx: self.angle_idx,
        }
    }

    pub(crate) fn from_peak(level: usize, angle_deg: f32, peak: Peak) -> Self {
        Self {
            level,
            x: peak.x,
            y: peak.y,
            angle_idx: peak.angle_idx,
            angle_deg,
            score: peak.score,
        }
    }
}

fn upscale_pos(x: usize, y: usize) -> (usize, usize) {
    (x.saturating_mul(2), y.saturating_mul(2))
}

fn roi_bounds(
    x: usize,
    y: usize,
    radius: usize,
    max_x: usize,
    max_y: usize,
) -> Option<(usize, usize, usize, usize)> {
    let mut x0 = x.saturating_sub(radius);
    let mut y0 = y.saturating_sub(radius);
    let mut x1 = x.saturating_add(radius);
    let mut y1 = y.saturating_add(radius);

    if x0 > max_x || y0 > max_y {
        return None;
    }

    x0 = x0.min(max_x);
    y0 = y0.min(max_y);
    x1 = x1.min(max_x);
    y1 = y1.min(max_y);

    if x0 > x1 || y0 > y1 {
        return None;
    }

    Some((x0, y0, x1, y1))
}

pub(crate) fn refine_to_finer_level(
    image: ImageView<'_, u8>,
    compiled: &CompiledTemplate,
    finer_level: usize,
    prev: &[Candidate],
    cfg: &MatchConfig,
) -> CorrMatchResult<Vec<Candidate>> {
    if prev.is_empty() {
        return Ok(Vec::new());
    }

    let grid = compiled
        .angle_grid(finer_level)
        .ok_or(CorrMatchError::IndexOutOfBounds {
            index: finer_level,
            len: compiled.num_levels(),
            context: "level",
        })?;
    let (tpl_width, tpl_height) =
        compiled
            .level_size(finer_level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: finer_level,
                len: compiled.num_levels(),
                context: "level",
            })?;

    let img_width = image.width();
    let img_height = image.height();
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
    let mut all_peaks = Vec::new();

    for cand in prev.iter().copied() {
        debug_assert!(cand.level > finer_level);
        let (x_up, y_up) = upscale_pos(cand.x, cand.y);
        let roi = match roi_bounds(x_up, y_up, cfg.roi_radius, max_x, max_y) {
            Some(bounds) => bounds,
            None => continue,
        };

        let half_range = cfg.angle_half_range_steps as f32 * grid.step_deg();
        let angle_indices = grid.indices_within(cand.angle_deg, half_range);
        for angle_idx in angle_indices {
            let rotated = compiled.rotated(finer_level, angle_idx)?;
            let plan = rotated.plan();
            let peaks = scan_masked_zncc_scalar_roi(
                image,
                plan,
                angle_idx,
                roi.0,
                roi.1,
                roi.2,
                roi.3,
                cfg.per_angle_topk,
                cfg.min_var_i,
                cfg.min_score,
            )?;
            all_peaks.extend(peaks);
        }
    }

    if all_peaks.is_empty() {
        return Ok(Vec::new());
    }

    let mut kept = crate::nms_2d(&mut all_peaks, cfg.nms_radius);
    if kept.len() > cfg.beam_width {
        kept.truncate(cfg.beam_width);
    }

    let mut out = Vec::with_capacity(kept.len());
    for peak in kept.drain(..) {
        let angle_deg = grid.angle_at(peak.angle_idx);
        out.push(Candidate::from_peak(finer_level, angle_deg, peak));
    }

    Ok(out)
}
