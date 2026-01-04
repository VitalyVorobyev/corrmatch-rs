//! Image pyramid construction for grayscale `u8` images.
//!
//! Downsampling uses a 2x2 box filter with integer rounding:
//! `dst = ((a + b + c + d) + 2) / 4`. This is a deterministic baseline
//! suitable for early scaffolding without introducing blur kernels yet.

use crate::image::{ImageView, OwnedImage};
use crate::util::{CorrMatchError, CorrMatchResult};

/// Owned image pyramid built from a base level.
pub struct ImagePyramid {
    levels: Vec<OwnedImage>,
}

impl ImagePyramid {
    /// Builds a pyramid from a base grayscale view.
    ///
    /// `max_levels` is clamped to at least 1 so the base level is always present.
    pub fn build_u8(base: ImageView<'_, u8>, max_levels: usize) -> CorrMatchResult<Self> {
        let max_levels = max_levels.max(1);
        let mut levels = Vec::new();
        levels.push(OwnedImage::from_view(base)?);

        while levels.len() < max_levels {
            let prev = levels.last().expect("levels is not empty");
            let src = prev.view();
            if src.width() < 2 || src.height() < 2 {
                break;
            }

            let dst_width = src.width() / 2;
            let dst_height = src.height() / 2;
            let dst_len =
                dst_width
                    .checked_mul(dst_height)
                    .ok_or(CorrMatchError::InvalidDimensions {
                        width: dst_width,
                        height: dst_height,
                    })?;
            let mut dst = vec![0u8; dst_len];

            for y in 0..dst_height {
                let row0 = src.row(y * 2).ok_or_else(|| {
                    let needed = (y * 2 + 1)
                        .checked_mul(src.stride())
                        .and_then(|v| v.checked_add(src.width()))
                        .unwrap_or(usize::MAX);
                    CorrMatchError::BufferTooSmall {
                        needed,
                        got: src.as_slice().len(),
                    }
                })?;
                let row1 = src.row(y * 2 + 1).ok_or_else(|| {
                    let needed = (y * 2 + 2)
                        .checked_mul(src.stride())
                        .and_then(|v| v.checked_add(src.width()))
                        .unwrap_or(usize::MAX);
                    CorrMatchError::BufferTooSmall {
                        needed,
                        got: src.as_slice().len(),
                    }
                })?;

                for x in 0..dst_width {
                    let a = row0[2 * x];
                    let b = row0[2 * x + 1];
                    let c = row1[2 * x];
                    let d = row1[2 * x + 1];
                    let sum = u16::from(a) + u16::from(b) + u16::from(c) + u16::from(d);
                    dst[y * dst_width + x] = ((sum + 2) / 4) as u8;
                }
            }

            levels.push(OwnedImage::new(dst, dst_width, dst_height)?);
        }

        Ok(Self { levels })
    }

    /// Returns all pyramid levels (level 0 is the base resolution).
    pub fn levels(&self) -> &[OwnedImage] {
        &self.levels
    }

    /// Returns a view for a specific pyramid level.
    pub fn level(&self, index: usize) -> Option<ImageView<'_, u8>> {
        self.levels.get(index).map(|level| level.view())
    }

    pub(crate) fn into_levels(self) -> Vec<OwnedImage> {
        self.levels
    }
}
