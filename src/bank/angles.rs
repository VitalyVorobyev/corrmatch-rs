//! Angle grid generation and lookup for rotation search.

use crate::util::math::wrap_deg;
use crate::util::{CorrMatchError, CorrMatchResult};

/// Discrete, circular angle grid in degrees.
#[derive(Clone, Debug)]
pub struct AngleGrid {
    min_deg: f32,
    max_deg: f32,
    step_deg: f32,
    len: usize,
}

impl AngleGrid {
    /// Creates a full grid over [-180, 180).
    pub fn full(step_deg: f32) -> CorrMatchResult<Self> {
        Self::new(-180.0, 180.0, step_deg)
    }

    /// Creates a grid over `[min_deg, max_deg)` with a positive step.
    pub fn new(min_deg: f32, max_deg: f32, step_deg: f32) -> CorrMatchResult<Self> {
        if !min_deg.is_finite() || !max_deg.is_finite() || !step_deg.is_finite() {
            return Err(CorrMatchError::InvalidAngleGrid {
                reason: "non-finite angle grid parameters",
            });
        }
        if step_deg <= 0.0 {
            return Err(CorrMatchError::InvalidAngleGrid {
                reason: "step_deg must be > 0",
            });
        }
        if max_deg <= min_deg {
            return Err(CorrMatchError::InvalidAngleGrid {
                reason: "max_deg must be greater than min_deg",
            });
        }

        let mut len = 0usize;
        loop {
            let angle = min_deg + (len as f32) * step_deg;
            if angle >= max_deg {
                break;
            }
            len += 1;
        }
        if len == 0 {
            return Err(CorrMatchError::InvalidAngleGrid {
                reason: "angle grid produced no samples",
            });
        }

        Ok(Self {
            min_deg,
            max_deg,
            step_deg,
            len,
        })
    }

    /// Returns the number of discrete angles in the grid.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the grid has no angles.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the minimum angle in degrees (inclusive).
    pub fn min_deg(&self) -> f32 {
        self.min_deg
    }

    /// Returns the maximum angle in degrees (exclusive).
    pub fn max_deg(&self) -> f32 {
        self.max_deg
    }

    /// Returns the grid step size in degrees.
    pub fn step_deg(&self) -> f32 {
        self.step_deg
    }

    /// Returns the wrapped angle for the given index.
    pub fn angle_at(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.len);
        wrap_deg(self.min_deg + (idx as f32) * self.step_deg)
    }

    /// Iterates over all angles in the grid.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.len).map(|idx| self.angle_at(idx))
    }

    /// Returns the nearest index to `angle_deg` using circular distance.
    pub fn nearest_index(&self, angle_deg: f32) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (idx, angle) in self.iter().enumerate() {
            let dist = wrap_deg(angle_deg - angle).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Returns indices within `half_range_deg` of `center_deg` using circular distance.
    pub fn indices_within(&self, center_deg: f32, half_range_deg: f32) -> Vec<usize> {
        if half_range_deg < 0.0 {
            return Vec::new();
        }
        let mut indices = Vec::new();
        for (idx, angle) in self.iter().enumerate() {
            let dist = wrap_deg(angle - center_deg).abs();
            if dist <= half_range_deg {
                indices.push(idx);
            }
        }
        indices
    }
}
