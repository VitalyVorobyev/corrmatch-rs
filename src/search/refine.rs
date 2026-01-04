//! Refinement search around coarse candidates.
//!
//! Refinement upsamples coarse candidates to finer levels and searches a local
//! ROI and angle neighborhood to improve position and angle estimates.

use crate::bank::CompiledTemplate;
use crate::candidate::topk::Peak;
use crate::refine::quad1d::quad_peak_offset_1d;
use crate::refine::quad2d::refine_subpixel_2d;
use crate::search::scan::{scan_masked_zncc_scalar_roi, score_masked_zncc_at};
use crate::search::{Match, MatchConfig};
use crate::util::math::wrap_deg;
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

/// Refines the best candidate at the finest level with subpixel and subangle fits.
pub(crate) fn refine_final_match(
    image: ImageView<'_, u8>,
    compiled: &CompiledTemplate,
    level: usize,
    best: Candidate,
    cfg: &MatchConfig,
) -> CorrMatchResult<Match> {
    let grid = compiled
        .angle_grid(level)
        .ok_or(CorrMatchError::IndexOutOfBounds {
            index: level,
            len: compiled.num_levels(),
            context: "level",
        })?;
    let (tpl_width, tpl_height) =
        compiled
            .level_size(level)
            .ok_or(CorrMatchError::IndexOutOfBounds {
                index: level,
                len: compiled.num_levels(),
                context: "level",
            })?;

    let img_width = image.width();
    let img_height = image.height();
    if img_width < tpl_width || img_height < tpl_height {
        return Err(CorrMatchError::RoiOutOfBounds {
            x: best.x,
            y: best.y,
            width: tpl_width,
            height: tpl_height,
            img_width,
            img_height,
        });
    }
    let max_x = img_width - tpl_width;
    let max_y = img_height - tpl_height;
    if best.x > max_x || best.y > max_y {
        return Err(CorrMatchError::RoiOutOfBounds {
            x: best.x,
            y: best.y,
            width: tpl_width,
            height: tpl_height,
            img_width,
            img_height,
        });
    }

    let plan = compiled.rotated(level, best.angle_idx)?.plan();

    let mut s = [[f32::NEG_INFINITY; 3]; 3];
    let offsets = [-1isize, 0, 1];
    for (iy, &dy) in offsets.iter().enumerate() {
        let y = best.y as isize + dy;
        if y < 0 || y > max_y as isize {
            continue;
        }
        for (ix, &dx) in offsets.iter().enumerate() {
            let x = best.x as isize + dx;
            if x < 0 || x > max_x as isize {
                continue;
            }
            s[iy][ix] = score_masked_zncc_at(image, plan, x as usize, y as usize, cfg.min_var_i);
        }
    }

    let center_score = if s[1][1].is_finite() {
        s[1][1]
    } else {
        best.score
    };
    let (x_ref, y_ref) = refine_subpixel_2d(best.x, best.y, s);

    let len = grid.len();
    debug_assert!(len > 0);
    let center_angle = grid.angle_at(best.angle_idx);
    let step = grid.step_deg();
    let im = (best.angle_idx + len - 1) % len;
    let ip = (best.angle_idx + 1) % len;
    let sm = score_masked_zncc_at(
        image,
        compiled.rotated(level, im)?.plan(),
        best.x,
        best.y,
        cfg.min_var_i,
    );
    let sp = score_masked_zncc_at(
        image,
        compiled.rotated(level, ip)?.plan(),
        best.x,
        best.y,
        cfg.min_var_i,
    );
    let angle_offset = quad_peak_offset_1d(sm, center_score, sp).unwrap_or(0.0);
    let angle_deg = wrap_deg(center_angle + angle_offset * step);

    Ok(Match {
        x: x_ref,
        y: y_ref,
        angle_deg,
        score: center_score,
    })
}
