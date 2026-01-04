//! Top-K candidate tracking for match peaks.

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

        let mut min_idx = 0usize;
        let mut min_score = self.items[0].score;
        for (idx, item) in self.items.iter().enumerate().skip(1) {
            if item.score < min_score {
                min_score = item.score;
                min_idx = idx;
            }
        }

        if peak.score > min_score {
            self.items[min_idx] = peak;
        }
    }

    /// Returns peaks sorted by descending score.
    pub fn into_sorted_desc(mut self) -> Vec<Peak> {
        self.items
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        self.items
    }
}
