//! Template plan precomputation for ZNCC and SSD metrics.

use crate::image::ImageView;
use crate::util::{CorrMatchError, CorrMatchResult};
use std::sync::Arc;

/// Coordinate of a valid (unmasked) template pixel.
///
/// Using `u32` allows templates up to 2^32 − 1 pixels wide/tall,
/// which is sufficient for any foreseeable image size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidCoord {
    /// X coordinate within the template (column).
    pub x: u32,
    /// Y coordinate within the template (row).
    pub y: u32,
}

/// Precomputed statistics and zero-mean buffer for unmasked ZNCC matching.
pub struct TemplatePlan {
    width: usize,
    height: usize,
    mean: f32,
    inv_std: f32,
    var_t: f32,
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
        let var_t = (variance * count_f) as f32;
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
            var_t,
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

    /// Returns the template variance term used by ZNCC.
    pub fn var_t(&self) -> f32 {
        self.var_t
    }

    /// Returns the zero-mean template buffer in row-major order.
    pub fn zero_mean(&self) -> &[f32] {
        &self.zero_mean
    }
}

/// Precomputed template buffer for SSD matching.
pub struct SsdTemplatePlan {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl SsdTemplatePlan {
    /// Builds an SSD plan from a template view.
    pub fn from_view(tpl: ImageView<'_, u8>) -> CorrMatchResult<Self> {
        let width = tpl.width();
        let height = tpl.height();
        let count = width
            .checked_mul(height)
            .ok_or(CorrMatchError::InvalidDimensions { width, height })?;

        let mut data = Vec::with_capacity(count);
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
                data.push(value as f32);
            }
        }

        Ok(Self {
            width,
            height,
            data,
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

    /// Returns the template buffer in row-major order.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// Precomputed masked statistics for ZNCC-style matching on rotated templates.
pub struct MaskedTemplatePlan {
    width: usize,
    height: usize,
    sum_w: f32,
    var_t: f32,
    mask: Arc<[u8]>,
    angle_deg: f32,
    /// Precomputed coordinates where mask is non-zero (for branch-free iteration).
    valid_coords: Vec<ValidCoord>,
    /// Precomputed t_prime values at valid coordinates only.
    valid_t_prime: Vec<f32>,
}

impl MaskedTemplatePlan {
    /// Builds a masked plan from a rotated template view and a binary mask.
    pub fn from_rotated_u8(
        rot: ImageView<'_, u8>,
        mask: Vec<u8>,
        angle_deg: f32,
    ) -> CorrMatchResult<Self> {
        Self::from_rotated_parts(rot, Arc::from(mask), angle_deg)
    }

    pub(crate) fn from_rotated_parts(
        rot: ImageView<'_, u8>,
        mask: Arc<[u8]>,
        angle_deg: f32,
    ) -> CorrMatchResult<Self> {
        let width = rot.width();
        let height = rot.height();
        if width > u16::MAX as usize || height > u16::MAX as usize {
            return Err(CorrMatchError::InvalidDimensions { width, height });
        }
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

        let mut sum_w_count = 0usize;
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
                if mask[idx] != 0 {
                    sum_w_count += 1;
                    sum_wt += value as f32;
                }
            }
        }

        if sum_w_count == 0 {
            return Err(CorrMatchError::DegenerateTemplate {
                reason: "mask has no valid pixels",
            });
        }

        let sum_w = sum_w_count as f32;
        let mu_t = sum_wt / sum_w;
        let mut var_t = 0.0f32;
        let mut valid_coords = Vec::with_capacity(sum_w_count);
        let mut valid_t_prime = Vec::with_capacity(sum_w_count);
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
                if mask[idx] == 0 {
                    continue;
                }
                let value = value as f32 - mu_t;
                valid_coords.push(ValidCoord {
                    x: x as u32,
                    y: y as u32,
                });
                valid_t_prime.push(value);
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
            mask,
            angle_deg,
            valid_coords,
            valid_t_prime,
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

    /// Returns the binary mask buffer (0 or 1 per pixel).
    pub fn mask(&self) -> &[u8] {
        self.mask.as_ref()
    }

    /// Returns the rotation angle in degrees.
    pub fn angle_deg(&self) -> f32 {
        self.angle_deg
    }

    /// Returns precomputed coordinates where the mask is non-zero.
    ///
    /// Use with `valid_t_prime()` for branch-free iteration over valid pixels.
    pub fn valid_coords(&self) -> &[ValidCoord] {
        &self.valid_coords
    }

    /// Returns precomputed t_prime values at valid coordinates.
    ///
    /// This slice has the same length as `valid_coords()`.
    pub fn valid_t_prime(&self) -> &[f32] {
        &self.valid_t_prime
    }
}

/// Precomputed masked buffer for SSD matching on rotated templates.
pub struct MaskedSsdTemplatePlan {
    width: usize,
    height: usize,
    mask: Arc<[u8]>,
    angle_deg: f32,
    /// Precomputed coordinates where mask is non-zero (for branch-free iteration).
    valid_coords: Vec<ValidCoord>,
    /// Precomputed template data values at valid coordinates only.
    valid_data: Vec<f32>,
}

impl MaskedSsdTemplatePlan {
    /// Builds a masked SSD plan from a rotated template view and a binary mask.
    pub fn from_rotated_u8(
        rot: ImageView<'_, u8>,
        mask: Vec<u8>,
        angle_deg: f32,
    ) -> CorrMatchResult<Self> {
        Self::from_rotated_parts(rot, Arc::from(mask), angle_deg)
    }

    pub(crate) fn from_rotated_parts(
        rot: ImageView<'_, u8>,
        mask: Arc<[u8]>,
        angle_deg: f32,
    ) -> CorrMatchResult<Self> {
        let width = rot.width();
        let height = rot.height();
        if width > u16::MAX as usize || height > u16::MAX as usize {
            return Err(CorrMatchError::InvalidDimensions { width, height });
        }
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

        let mut sum_w = 0usize;
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
            for (x, &_value) in row.iter().enumerate() {
                let idx = y * width + x;
                if mask[idx] != 0 {
                    sum_w += 1;
                }
            }
        }

        if sum_w == 0 {
            return Err(CorrMatchError::DegenerateTemplate {
                reason: "mask has no valid pixels",
            });
        }

        // Precompute valid coordinates for branch-free iteration in hot loops.
        let mut valid_coords = Vec::with_capacity(sum_w);
        let mut valid_data = Vec::with_capacity(sum_w);
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
                if mask[idx] == 0 {
                    continue;
                }
                valid_coords.push(ValidCoord {
                    x: x as u32,
                    y: y as u32,
                });
                valid_data.push(value as f32);
            }
        }

        Ok(Self {
            width,
            height,
            mask,
            angle_deg,
            valid_coords,
            valid_data,
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

    /// Returns the binary mask buffer (0 or 1 per pixel).
    pub fn mask(&self) -> &[u8] {
        self.mask.as_ref()
    }

    /// Returns the rotation angle in degrees.
    pub fn angle_deg(&self) -> f32 {
        self.angle_deg
    }

    /// Returns precomputed coordinates where the mask is non-zero.
    ///
    /// Use with `valid_data()` for branch-free iteration over valid pixels.
    pub fn valid_coords(&self) -> &[ValidCoord] {
        &self.valid_coords
    }

    /// Returns precomputed template data values at valid coordinates.
    ///
    /// This slice has the same length as `valid_coords()`.
    pub fn valid_data(&self) -> &[f32] {
        &self.valid_data
    }
}
