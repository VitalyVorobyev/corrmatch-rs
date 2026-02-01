//! Summed-area tables for fast window statistics.

use crate::image::ImageView;
use crate::util::{CorrMatchError, CorrMatchResult};

/// Precomputed summed-area tables for intensity and squared intensity.
#[derive(Clone, Debug)]
pub(crate) struct IntegralImages {
    width: usize,
    height: usize,
    stride: usize,
    sum: Vec<f32>,
    sumsq: Vec<f32>,
}

impl IntegralImages {
    /// Builds integral images for a grayscale view.
    pub(crate) fn from_u8(view: ImageView<'_, u8>) -> CorrMatchResult<Self> {
        let width = view.width();
        let height = view.height();
        let stride = width
            .checked_add(1)
            .ok_or(CorrMatchError::InvalidDimensions { width, height })?;
        let len = (height + 1)
            .checked_mul(stride)
            .ok_or(CorrMatchError::InvalidDimensions { width, height })?;
        let mut sum = vec![0.0f32; len];
        let mut sumsq = vec![0.0f32; len];

        for y in 0..height {
            let row = view.row(y).ok_or_else(|| {
                let needed = (y + 1)
                    .checked_mul(view.stride())
                    .and_then(|v| v.checked_add(view.width()))
                    .unwrap_or(usize::MAX);
                CorrMatchError::BufferTooSmall {
                    needed,
                    got: view.as_slice().len(),
                }
            })?;
            let base = (y + 1) * stride;
            let prev = y * stride;
            let mut row_sum = 0.0f32;
            let mut row_sumsq = 0.0f32;
            for x in 0..width {
                let v = row[x] as f32;
                row_sum += v;
                row_sumsq += v * v;
                let idx = base + x + 1;
                sum[idx] = sum[prev + x + 1] + row_sum;
                sumsq[idx] = sumsq[prev + x + 1] + row_sumsq;
            }
        }

        Ok(Self {
            width,
            height,
            stride,
            sum,
            sumsq,
        })
    }

    /// Returns the sum of pixel intensities over a rectangle.
    #[inline]
    pub(crate) fn sum_rect(&self, x: usize, y: usize, width: usize, height: usize) -> f32 {
        let x1 = x + width;
        let y1 = y + height;
        let a = self.sum[y * self.stride + x];
        let b = self.sum[y * self.stride + x1];
        let c = self.sum[y1 * self.stride + x];
        let d = self.sum[y1 * self.stride + x1];
        d + a - b - c
    }

    /// Returns the sum of squared intensities over a rectangle.
    #[inline]
    pub(crate) fn sumsq_rect(&self, x: usize, y: usize, width: usize, height: usize) -> f32 {
        let x1 = x + width;
        let y1 = y + height;
        let a = self.sumsq[y * self.stride + x];
        let b = self.sumsq[y * self.stride + x1];
        let c = self.sumsq[y1 * self.stride + x];
        let d = self.sumsq[y1 * self.stride + x1];
        d + a - b - c
    }

    /// Returns the image width the integrals were built from.
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height the integrals were built from.
    pub(crate) fn height(&self) -> usize {
        self.height
    }
}
