//! Template plan precomputation for ZNCC-like metrics.

use crate::image::ImageView;
use crate::util::{CorrMatchError, CorrMatchResult};

/// Precomputed statistics and zero-mean buffer for template matching.
pub struct TemplatePlan {
    width: usize,
    height: usize,
    mean: f32,
    inv_std: f32,
    zero_mean: Vec<f32>,
}

impl TemplatePlan {
    /// Builds a plan from a template view.
    pub fn from_view(tpl: ImageView<'_, u8>) -> CorrMatchResult<Self> {
        let width = tpl.width();
        let height = tpl.height();
        let count = width
            .checked_mul(height)
            .ok_or(CorrMatchError::InvalidDimensions { width, height })?;

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for y in 0..height {
            let row = tpl.row(y).ok_or_else(|| {
                let needed = (y + 1)
                    .checked_mul(tpl.stride())
                    .and_then(|v| v.checked_add(tpl.width()))
                    .unwrap_or(usize::MAX);
                CorrMatchError::BufferTooSmall {
                    needed,
                    got: tpl.as_slice().len(),
                }
            })?;
            for &value in row {
                let v = value as f64;
                sum += v;
                sum_sq += v * v;
            }
        }

        let count_f = count as f64;
        let mean_f64 = sum / count_f;
        let variance = sum_sq / count_f - mean_f64 * mean_f64;
        if variance <= 1e-8 {
            return Err(CorrMatchError::DegenerateTemplate {
                reason: "zero variance",
            });
        }

        let mean = mean_f64 as f32;
        let inv_std = (1.0 / variance.sqrt()) as f32;
        let mut zero_mean = Vec::with_capacity(count);
        for y in 0..height {
            let row = tpl.row(y).ok_or_else(|| {
                let needed = (y + 1)
                    .checked_mul(tpl.stride())
                    .and_then(|v| v.checked_add(tpl.width()))
                    .unwrap_or(usize::MAX);
                CorrMatchError::BufferTooSmall {
                    needed,
                    got: tpl.as_slice().len(),
                }
            })?;
            for &value in row {
                zero_mean.push(value as f32 - mean);
            }
        }

        Ok(Self {
            width,
            height,
            mean,
            inv_std,
            zero_mean,
        })
    }

    /// Returns the template width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the template height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the mean intensity of the template.
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Returns the inverse standard deviation of the template.
    pub fn inv_std(&self) -> f32 {
        self.inv_std
    }

    /// Returns the zero-mean template buffer in row-major order.
    pub fn zero_mean(&self) -> &[f32] {
        &self.zero_mean
    }
}
