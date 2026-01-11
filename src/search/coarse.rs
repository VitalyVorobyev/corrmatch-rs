//! Coarse search over the pyramid and angle bank.
//!
//! Coarse search evaluates the full translation range at the coarsest level
//! for each discrete rotation angle, then merges and prunes candidates.

use crate::bank::CompiledTemplate;
use crate::candidate::nms::nms_2d;
#[cfg(feature = "rayon")]
use crate::kernel::rayon::{ssd_unmasked_scan_full_par, zncc_unmasked_scan_full_par};
use crate::kernel::scalar::{SsdMaskedScalar, ZnccMaskedScalar};
use crate::kernel::{Kernel, ScanParams};
use crate::search::refine::Candidate;
use crate::search::{MatchConfig, Metric};
use crate::trace::{trace_event, trace_span};
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Kernel type aliases for unmasked kernels - use SIMD when available
#[cfg(not(feature = "simd"))]
use crate::kernel::scalar::{SsdUnmaskedScalar as SsdUnmasked, ZnccUnmaskedScalar as ZnccUnmasked};
#[cfg(feature = "simd")]
use crate::kernel::simd::{SsdUnmaskedSimd as SsdUnmasked, ZnccUnmaskedSimd as ZnccUnmasked};

pub(crate) fn coarse_search_level(
    image: ImageView<'_, u8>,
    compiled: &CompiledTemplate,
    level: usize,
    cfg: &MatchConfig,
) -> CorrMatchResult<Vec<Candidate>> {
    let grid = compiled
        .angle_grid(level)
        .ok_or(CorrMatchError::IndexOutOfBounds {
            index: level,
            len: compiled.num_levels(),
            context: "level",
        })?;

    let _span = trace_span!("coarse_search", level = level, angles = grid.len()).entered();

    let params = ScanParams {
        topk: cfg.per_angle_topk,
        min_var_i: cfg.min_var_i,
        min_score: cfg.min_score,
    };
    let mut all_candidates = Vec::new();
    for angle_idx in 0..grid.len() {
        let peaks = match cfg.metric {
            Metric::Zncc => {
                let plan = compiled.rotated_zncc_plan(level, angle_idx)?;
                <ZnccMaskedScalar as Kernel>::scan_full(image, plan, angle_idx, params)?
            }
            Metric::Ssd => {
                let plan = compiled.rotated_ssd_plan(level, angle_idx)?;
                <SsdMaskedScalar as Kernel>::scan_full(image, plan, angle_idx, params)?
            }
        };
        for peak in peaks {
            let angle_deg = grid.angle_at(peak.angle_idx);
            all_candidates.push(Candidate::from_peak(level, angle_deg, peak));
        }
    }

    if all_candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut peaks: Vec<_> = all_candidates
        .iter()
        .copied()
        .map(Candidate::to_peak)
        .collect();
    let mut kept = nms_2d(&mut peaks, cfg.nms_radius);
    if kept.len() > cfg.beam_width {
        kept.truncate(cfg.beam_width);
    }

    let mut out = Vec::with_capacity(kept.len());
    for peak in kept.drain(..) {
        let angle_deg = grid.angle_at(peak.angle_idx);
        out.push(Candidate::from_peak(level, angle_deg, peak));
    }

    trace_event!("coarse_candidates", count = out.len());
    Ok(out)
}

/// Coarse search without rotation using an unmasked kernel.
pub(crate) fn coarse_search_level_unmasked(
    image: ImageView<'_, u8>,
    compiled: &CompiledTemplate,
    level: usize,
    cfg: &MatchConfig,
) -> CorrMatchResult<Vec<Candidate>> {
    let _span = trace_span!("coarse_search", level = level, angles = 1).entered();

    let params = ScanParams {
        topk: cfg.per_angle_topk,
        min_var_i: cfg.min_var_i,
        min_score: cfg.min_score,
    };
    let mut peaks = match cfg.metric {
        Metric::Zncc => {
            let plan = compiled.unmasked_zncc_plan(level)?;
            <ZnccUnmasked as Kernel>::scan_full(image, plan, 0, params)?
        }
        Metric::Ssd => {
            let plan = compiled.unmasked_ssd_plan(level)?;
            <SsdUnmasked as Kernel>::scan_full(image, plan, 0, params)?
        }
    };
    if peaks.is_empty() {
        return Ok(Vec::new());
    }

    let mut kept = nms_2d(&mut peaks, cfg.nms_radius);
    if kept.len() > cfg.beam_width {
        kept.truncate(cfg.beam_width);
    }

    let mut out = Vec::with_capacity(kept.len());
    for peak in kept.drain(..) {
        out.push(Candidate::from_peak(level, 0.0, peak));
    }

    trace_event!("coarse_candidates", count = out.len());
    Ok(out)
}

/// Coarse search over angles in parallel (rayon).
#[cfg(feature = "rayon")]
pub(crate) fn coarse_search_level_par(
    image: ImageView<'_, u8>,
    compiled: &CompiledTemplate,
    level: usize,
    cfg: &MatchConfig,
) -> CorrMatchResult<Vec<Candidate>> {
    let grid = compiled
        .angle_grid(level)
        .ok_or(CorrMatchError::IndexOutOfBounds {
            index: level,
            len: compiled.num_levels(),
            context: "level",
        })?;

    let _span = trace_span!(
        "coarse_search",
        level = level,
        angles = grid.len(),
        parallel = true
    )
    .entered();

    let params = ScanParams {
        topk: cfg.per_angle_topk,
        min_var_i: cfg.min_var_i,
        min_score: cfg.min_score,
    };
    let results: Vec<_> = (0..grid.len())
        .into_par_iter()
        .map(|angle_idx| match cfg.metric {
            Metric::Zncc => {
                let plan = compiled.rotated_zncc_plan(level, angle_idx)?;
                <ZnccMaskedScalar as Kernel>::scan_full(image, plan, angle_idx, params)
            }
            Metric::Ssd => {
                let plan = compiled.rotated_ssd_plan(level, angle_idx)?;
                <SsdMaskedScalar as Kernel>::scan_full(image, plan, angle_idx, params)
            }
        })
        .collect();

    let mut peaks = Vec::new();
    for result in results {
        peaks.extend(result?);
    }
    if peaks.is_empty() {
        return Ok(Vec::new());
    }

    let mut kept = nms_2d(&mut peaks, cfg.nms_radius);
    if kept.len() > cfg.beam_width {
        kept.truncate(cfg.beam_width);
    }

    let mut out = Vec::with_capacity(kept.len());
    for peak in kept.drain(..) {
        let angle_deg = grid.angle_at(peak.angle_idx);
        out.push(Candidate::from_peak(level, angle_deg, peak));
    }

    trace_event!("coarse_candidates", count = out.len());
    Ok(out)
}

/// Coarse search without rotation using an unmasked kernel (parallel).
///
/// Uses row-parallel scanning to distribute work across threads.
#[cfg(feature = "rayon")]
pub(crate) fn coarse_search_level_unmasked_par(
    image: ImageView<'_, u8>,
    compiled: &CompiledTemplate,
    level: usize,
    cfg: &MatchConfig,
) -> CorrMatchResult<Vec<Candidate>> {
    let _span = trace_span!("coarse_search", level = level, angles = 1, parallel = true).entered();

    let params = ScanParams {
        topk: cfg.per_angle_topk,
        min_var_i: cfg.min_var_i,
        min_score: cfg.min_score,
    };
    // Use row-parallel scan functions for actual parallelization
    let mut peaks = match cfg.metric {
        Metric::Zncc => {
            let plan = compiled.unmasked_zncc_plan(level)?;
            zncc_unmasked_scan_full_par(image, plan, 0, params)?
        }
        Metric::Ssd => {
            let plan = compiled.unmasked_ssd_plan(level)?;
            ssd_unmasked_scan_full_par(image, plan, 0, params)?
        }
    };
    if peaks.is_empty() {
        return Ok(Vec::new());
    }

    let mut kept = nms_2d(&mut peaks, cfg.nms_radius);
    if kept.len() > cfg.beam_width {
        kept.truncate(cfg.beam_width);
    }

    let mut out = Vec::with_capacity(kept.len());
    for peak in kept.drain(..) {
        out.push(Candidate::from_peak(level, 0.0, peak));
    }

    trace_event!("coarse_candidates", count = out.len());
    Ok(out)
}
