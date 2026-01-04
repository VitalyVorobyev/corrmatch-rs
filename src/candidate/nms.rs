//! Non-maximum suppression utilities for candidates.

use crate::candidate::topk::Peak;

/// Applies 2D non-maximum suppression using Chebyshev distance.
///
/// Peaks are sorted by descending score and kept if they are farther than
/// `radius` in Chebyshev distance from all previously kept peaks.
pub fn nms_2d(peaks: &mut [Peak], radius: usize) -> Vec<Peak> {
    if radius == 0 {
        peaks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        return peaks.to_owned();
    }

    peaks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
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
