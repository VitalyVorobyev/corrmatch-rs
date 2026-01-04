//! Rotation handling and angle normalization utilities.

use crate::image::{ImageView, OwnedImage};
use crate::util::math::sin_cos_deg;

/// Rotates a grayscale template using bilinear sampling.
///
/// Rotation is performed about the image center with
/// `cx = (w - 1) / 2` and `cy = (h - 1) / 2` in floating-point coordinates.
/// Each destination pixel center `(x, y)` is mapped to the source coordinate
/// using inverse rotation. Samples outside the source bounds are filled with
/// `fill`. Bilinear interpolation clamps indices to the valid image range.
/// The output image has the same dimensions as the input and uses rounding
/// to the nearest integer before clamping to `[0, 255]`.
pub fn rotate_u8_bilinear(src: ImageView<'_, u8>, angle_deg: f32, fill: u8) -> OwnedImage {
    let width = src.width();
    let height = src.height();
    let mut out = vec![fill; width * height];

    let (sin_a, cos_a) = sin_cos_deg(angle_deg);
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let max_x = width as f32 - 1.0;
    let max_y = height as f32 - 1.0;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let src_x = cos_a * dx + sin_a * dy + cx;
            let src_y = -sin_a * dx + cos_a * dy + cy;

            let epsilon = 1e-6;
            if !src_x.is_finite()
                || !src_y.is_finite()
                || src_x < -epsilon
                || src_y < -epsilon
                || src_x > max_x + epsilon
                || src_y > max_y + epsilon
            {
                out[y * width + x] = fill;
                continue;
            }

            let src_x = src_x.clamp(0.0, max_x);
            let src_y = src_y.clamp(0.0, max_y);
            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let row0 = src.row(y0).expect("row in bounds");
            let row1 = src.row(y1).expect("row in bounds");
            let a = row0[x0] as f32;
            let b = row0[x1] as f32;
            let c = row1[x0] as f32;
            let d = row1[x1] as f32;

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;
            let value = a * w00 + b * w10 + c * w01 + d * w11;

            let rounded = value.round().clamp(0.0, 255.0);
            out[y * width + x] = rounded as u8;
        }
    }

    OwnedImage::new(out, width, height).expect("rotation output is contiguous")
}
