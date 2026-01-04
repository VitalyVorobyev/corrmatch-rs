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

/// Precomputed masked statistics for ZNCC-style matching on rotated templates.
pub struct MaskedTemplatePlan {
    width: usize,
    height: usize,
    sum_w: f32,
    var_t: f32,
    t_prime: Vec<f32>,
    mask: Vec<u8>,
    angle_deg: f32,
}

impl MaskedTemplatePlan {
    /// Builds a masked plan from a rotated template view and a binary mask.
    pub fn from_rotated_u8(
        rot: ImageView<'_, u8>,
        mask: Vec<u8>,
        angle_deg: f32,
    ) -> CorrMatchResult<Self> {
        let width = rot.width();
        let height = rot.height();
        let needed = width
            .checked_mul(height)
            .ok_or(CorrMatchError::InvalidDimensions { width, height })?;
        if mask.len() < needed {
            return Err(CorrMatchError::BufferTooSmall {
                needed,
                got: mask.len(),
            });
        }
        if mask.len() > needed {
            return Err(CorrMatchError::InvalidDimensions { width, height });
        }

        let mut sum_w = 0.0f32;
        let mut sum_wt = 0.0f32;
        for y in 0..height {
            let row = rot.row(y).ok_or_else(|| {
                let needed = (y + 1)
                    .checked_mul(rot.stride())
                    .and_then(|v| v.checked_add(rot.width()))
                    .unwrap_or(usize::MAX);
                CorrMatchError::BufferTooSmall {
                    needed,
                    got: rot.as_slice().len(),
                }
            })?;
            for (x, &value) in row.iter().enumerate() {
                let idx = y * width + x;
                let w = if mask[idx] == 0 { 0.0 } else { 1.0 };
                let t = value as f32;
                sum_w += w;
                sum_wt += w * t;
            }
        }

        if sum_w < 1.0 {
            return Err(CorrMatchError::DegenerateTemplate {
                reason: "mask has no valid pixels",
            });
        }

        let mu_t = sum_wt / sum_w;
        let mut t_prime = Vec::with_capacity(needed);
        let mut var_t = 0.0f32;
        for y in 0..height {
            let row = rot.row(y).ok_or_else(|| {
                let needed = (y + 1)
                    .checked_mul(rot.stride())
                    .and_then(|v| v.checked_add(rot.width()))
                    .unwrap_or(usize::MAX);
                CorrMatchError::BufferTooSmall {
                    needed,
                    got: rot.as_slice().len(),
                }
            })?;
            for (x, &value) in row.iter().enumerate() {
                let idx = y * width + x;
                let w = if mask[idx] == 0 { 0.0 } else { 1.0 };
                let value = w * (value as f32 - mu_t);
                t_prime.push(value);
                var_t += value * value;
            }
        }

        if var_t <= 1e-8 {
            return Err(CorrMatchError::DegenerateTemplate {
                reason: "template variance too small",
            });
        }

        Ok(Self {
            width,
            height,
            sum_w,
            var_t,
            t_prime,
            mask,
            angle_deg,
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

    /// Returns the sum of mask weights (count of valid pixels).
    pub fn sum_w(&self) -> f32 {
        self.sum_w
    }

    /// Returns the template variance term used by ZNCC.
    pub fn var_t(&self) -> f32 {
        self.var_t
    }

    /// Returns the masked zero-mean template buffer.
    pub fn t_prime(&self) -> &[f32] {
        &self.t_prime
    }

    /// Returns the binary mask buffer (0 or 1 per pixel).
    pub fn mask(&self) -> &[u8] {
        &self.mask
    }

    /// Returns the rotation angle in degrees.
    pub fn angle_deg(&self) -> f32 {
        self.angle_deg
    }
}
