//! Top-K candidate tracking for match peaks.

use std::cmp::Ordering;

/// Peak candidate in image space for a specific rotation angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// X coordinate (column) of the peak.
    pub x: usize,
    /// Y coordinate (row) of the peak.
    pub y: usize,
    /// ZNCC score at the peak location.
    pub score: f32,
    /// Index into the angle grid.
    pub angle_idx: usize,
}

fn peak_cmp_desc(a: &Peak, b: &Peak) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.y.cmp(&b.y))
        .then_with(|| a.x.cmp(&b.x))
        .then_with(|| a.angle_idx.cmp(&b.angle_idx))
}

/// Sorts peaks by descending score with deterministic tie-breaking.
pub(crate) fn sort_peaks_desc(peaks: &mut [Peak]) {
    peaks.sort_by(peak_cmp_desc);
}

/// Top-K container with O(k) insertion cost.
pub struct TopK<T> {
    k: usize,
    items: Vec<T>,
}

impl TopK<Peak> {
    /// Creates a new Top-K collector.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k),
        }
    }

    /// Pushes a peak, evicting the lowest score if at capacity.
    pub fn push(&mut self, peak: Peak) {
        if self.k == 0 {
            return;
        }
        if self.items.len() < self.k {
            self.items.push(peak);
            return;
        }

        let mut worst_idx = 0usize;
        for (idx, item) in self.items.iter().enumerate().skip(1) {
            if peak_cmp_desc(item, &self.items[worst_idx]) == Ordering::Greater {
                worst_idx = idx;
            }
        }

        if peak_cmp_desc(&peak, &self.items[worst_idx]) == Ordering::Less {
            self.items[worst_idx] = peak;
        }
    }

    /// Returns peaks sorted by descending score.
    pub fn into_sorted_desc(mut self) -> Vec<Peak> {
        sort_peaks_desc(&mut self.items);
        self.items
    }
}
