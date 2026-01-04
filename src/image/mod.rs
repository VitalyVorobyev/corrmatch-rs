//! Image views and pyramid utilities.
//!
//! `ImageView` is a borrowed 2D view into a 1D buffer with an explicit stride.
//! The stride counts elements between the starts of consecutive rows, so a
//! stride larger than the width represents padded rows. ROI slices are zero-copy
//! views into the same backing slice and retain the original stride.

use crate::util::{CorrMatchError, CorrMatchResult};

pub mod pyramid;

/// Borrowed 2D image view with an explicit stride.
#[derive(Copy, Clone)]
pub struct ImageView<'a, T> {
    data: &'a [T],
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a, T> ImageView<'a, T> {
    /// Creates a contiguous view with `stride == width`.
    pub fn from_slice(data: &'a [T], width: usize, height: usize) -> CorrMatchResult<Self> {
        Self::new(data, width, height, width)
    }

    /// Creates a view with an explicit stride.
    pub fn new(data: &'a [T], width: usize, height: usize, stride: usize) -> CorrMatchResult<Self> {
        let needed = required_len(width, height, stride)?;
        if data.len() < needed {
            return Err(CorrMatchError::BufferTooSmall {
                needed,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }

    /// Returns the image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the stride in elements between row starts.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the backing slice including any row padding.
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// Returns the element at `(x, y)` if it is within bounds.
    pub fn get(&self, x: usize, y: usize) -> Option<&'a T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = y.checked_mul(self.stride)?.checked_add(x)?;
        self.data.get(idx)
    }

    /// Returns a contiguous slice for row `y` with length `width`.
    pub fn row(&self, y: usize) -> Option<&'a [T]> {
        if y >= self.height {
            return None;
        }
        let start = y.checked_mul(self.stride)?;
        let end = start.checked_add(self.width)?;
        self.data.get(start..end)
    }

    /// Returns a zero-copy ROI view into the same backing buffer.
    pub fn roi(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> CorrMatchResult<ImageView<'a, T>> {
        if width == 0 || height == 0 {
            return Err(CorrMatchError::InvalidDimensions { width, height });
        }

        let img_width = self.width;
        let img_height = self.height;
        if x >= img_width || y >= img_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x,
                y,
                width,
                height,
                img_width,
                img_height,
            });
        }

        let end_x = x.checked_add(width).ok_or(CorrMatchError::RoiOutOfBounds {
            x,
            y,
            width,
            height,
            img_width,
            img_height,
        })?;
        let end_y = y
            .checked_add(height)
            .ok_or(CorrMatchError::RoiOutOfBounds {
                x,
                y,
                width,
                height,
                img_width,
                img_height,
            })?;
        if end_x > img_width || end_y > img_height {
            return Err(CorrMatchError::RoiOutOfBounds {
                x,
                y,
                width,
                height,
                img_width,
                img_height,
            });
        }

        let start = y
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(x))
            .ok_or(CorrMatchError::InvalidDimensions {
                width: img_width,
                height: img_height,
            })?;
        let data = self
            .data
            .get(start..)
            .ok_or(CorrMatchError::BufferTooSmall {
                needed: start.saturating_add(1),
                got: self.data.len(),
            })?;

        ImageView::new(data, width, height, self.stride)
    }
}

fn required_len(width: usize, height: usize, stride: usize) -> CorrMatchResult<usize> {
    if width == 0 || height == 0 {
        return Err(CorrMatchError::InvalidDimensions { width, height });
    }
    if stride < width {
        return Err(CorrMatchError::InvalidStride { width, stride });
    }
    let needed = (height - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(width))
        .ok_or(CorrMatchError::InvalidDimensions { width, height })?;
    Ok(needed)
}
