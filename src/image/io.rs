//! Convenience helpers for loading images via the `image` crate.
//!
//! Available when the `image-io` feature is enabled.

use crate::image::{ImageView, OwnedImage};
use crate::util::{CorrMatchError, CorrMatchResult};
use std::path::Path;

/// Creates a borrowed view from a grayscale image buffer.
pub fn view_from_gray_image(img: &image::GrayImage) -> CorrMatchResult<ImageView<'_, u8>> {
    let width = img.width() as usize;
    let height = img.height() as usize;
    ImageView::from_slice(img.as_raw(), width, height)
}

/// Creates an owned image from a grayscale image buffer.
pub fn owned_from_gray_image(img: &image::GrayImage) -> CorrMatchResult<OwnedImage> {
    let width = img.width() as usize;
    let height = img.height() as usize;
    OwnedImage::new(img.as_raw().clone(), width, height)
}

/// Creates an owned grayscale image from a dynamic image.
pub fn owned_from_dynamic_image(img: &image::DynamicImage) -> CorrMatchResult<OwnedImage> {
    let gray = img.to_luma8();
    owned_from_gray_image(&gray)
}

/// Loads an image from disk and converts it to a grayscale owned image.
pub fn load_gray_image<P: AsRef<Path>>(path: P) -> CorrMatchResult<OwnedImage> {
    let img = image::open(path).map_err(|err| CorrMatchError::ImageIo {
        reason: err.to_string(),
    })?;
    owned_from_dynamic_image(&img)
}
