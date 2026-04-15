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
///
/// # Boundary handling (unmasked path)
///
/// Source coordinates within `[-ε, w − 1 + ε] × [-ε, h − 1 + ε]` (with
/// `ε = 1e-6`) are accepted and clamped to `[0, w − 1] × [0, h − 1]` before
/// sampling. This tolerates floating-point rounding at the image edge (e.g. a
/// 180° rotation that produces `src_x = -1e-7` instead of `0.0`). Pixels
/// beyond this tolerance are set to `fill` without interpolation.
///
/// **Note**: this epsilon-clamp logic differs from [`rotate_u8_bilinear_masked`],
/// which uses strict bounds and additionally requires the 4-neighbour footprint
/// to be fully inside the source image. The masked variant deliberately drops
/// partial footprints to avoid silently blending in out-of-bounds values.
/// Prefer [`rotate_u8_bilinear_masked`] when the validity mask matters.
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

            // Skip pixels with out-of-bounds source coordinates.
            // Use small epsilon tolerance to handle floating-point imprecision
            // (e.g., 180 degree rotation can produce -1e-7 instead of 0.0).
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

            // Clamp to valid range before casting to usize.
            // This handles the epsilon-tolerance edge cases safely.
            let src_x = src_x.clamp(0.0, max_x);
            let src_y = src_y.clamp(0.0, max_y);
            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            // Invariant: src_x was clamped to [0.0, max_x] and src_y to [0.0, max_y],
            // where max_x = width - 1 and max_y = height - 1.  Therefore y0 ≤ max_y
            // and y1 = (y0 + 1).min(height - 1) ≤ height - 1, both within bounds.
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

    // Invariant: out has exactly width * height elements (allocated above), so
    // OwnedImage::new succeeds for any valid (width, height) pair.
    OwnedImage::new(out, width, height).expect("rotation output is contiguous")
}

/// Rotates a grayscale template using bilinear sampling and returns a validity mask.
///
/// Rotation is performed about the image center with
/// `cx = (w - 1) / 2` and `cy = (h - 1) / 2` in floating-point coordinates.
/// Each destination pixel center `(x, y)` is mapped to the source coordinate
/// using inverse rotation. A pixel is considered valid only if the source
/// coordinate is inside `[0, w - 1] × [0, h - 1]` **and** the 4-neighbor
/// footprint for bilinear sampling is fully inside bounds. Valid pixels are
/// marked with `mask = 1` and invalid pixels with `mask = 0`. Invalid pixels
/// are filled with `fill` without clamping to the image edge.
///
/// # Boundary handling (masked path)
///
/// This function uses **strict** bounds: the source coordinate must satisfy
/// `0.0 ≤ src_x ≤ w − 1` and `0.0 ≤ src_y ≤ h − 1`, and the 4-neighbour
/// footprint `[x0, x0+1] × [y0, y0+1]` must lie fully within the image.
/// Pixels failing either test are dropped (mask = 0). There is **no** epsilon
/// tolerance — a 180° rotation that produces `src_x = -1e-7` will be treated
/// as out-of-bounds.
///
/// **Contrast with [`rotate_u8_bilinear`]**: the unmasked path applies a
/// small epsilon tolerance (`1e-6`) and clamps coordinates, so it accepts
/// boundary-adjacent pixels that this function rejects. Use this masked
/// variant whenever the template validity mask is required (e.g. for
/// masked ZNCC kernels), because it never blends in out-of-bounds values.
pub fn rotate_u8_bilinear_masked(
    src: ImageView<'_, u8>,
    angle_deg: f32,
    fill: u8,
) -> (OwnedImage, Vec<u8>) {
    let width = src.width();
    let height = src.height();
    let mut out = vec![fill; width * height];
    let mut mask = vec![0u8; width * height];

    if width < 2 || height < 2 {
        // Invariant: out has width * height elements (vec![fill; width * height] above).
        return (
            OwnedImage::new(out, width, height).expect("rotation output is contiguous"),
            mask,
        );
    }

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

            if !src_x.is_finite() || !src_y.is_finite() || src_x < 0.0 || src_y < 0.0 {
                continue;
            }
            if src_x > max_x || src_y > max_y {
                continue;
            }

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            if x1 >= width || y1 >= height {
                continue;
            }

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;
            // Invariant: the `x1 >= width || y1 >= height` guard above ensures the full
            // 4-neighbour footprint is inside bounds, so y0 < y1 < height.
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

            let idx = y * width + x;
            out[idx] = value.round().clamp(0.0, 255.0) as u8;
            mask[idx] = 1;
        }
    }

    // Invariant: out and mask each have exactly width * height elements (allocated above).
    (
        OwnedImage::new(out, width, height).expect("rotation output is contiguous"),
        mask,
    )
}
