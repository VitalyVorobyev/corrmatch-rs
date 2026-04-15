//! Non-maximum suppression utilities for candidates.

use crate::candidate::topk::{sort_peaks_desc, Peak};

/// Applies 2D non-maximum suppression using Chebyshev distance.
///
/// Peaks are sorted by descending score and kept if they are farther than
/// `radius` in Chebyshev distance from all previously kept peaks.
pub fn nms_2d(peaks: &mut [Peak], radius: usize) -> Vec<Peak> {
    if radius == 0 {
        sort_peaks_desc(peaks);
        return peaks.to_owned();
    }

    sort_peaks_desc(peaks);
    let mut kept: Vec<Peak> = Vec::new();

    'outer: for peak in peaks.iter().copied() {
        for kept_peak in kept.iter() {
            let dx = peak.x.max(kept_peak.x) - peak.x.min(kept_peak.x);
            let dy = peak.y.max(kept_peak.y) - peak.y.min(kept_peak.y);
            let dist = dx.max(dy);
            if dist <= radius {
                continue 'outer;
            }
        }
        kept.push(peak);
    }

    kept
}
