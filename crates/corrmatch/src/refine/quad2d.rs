//! Quadratic 2D fitting for (x, y) refinement.

use crate::refine::quad1d::quad_peak_offset_1d;

/// Refines a subpixel peak using separable 1D quadratic fits.
///
/// The input is a 3x3 neighborhood of scores centered at `s[1][1]`. This
/// estimates `dx` from the center row and `dy` from the center column. The
/// returned coordinates are `center + offset`, falling back to the integer
/// center if the fit is ill-conditioned.
pub fn refine_subpixel_2d(center_x: usize, center_y: usize, s: [[f32; 3]; 3]) -> (f32, f32) {
    let dx = quad_peak_offset_1d(s[1][0], s[1][1], s[1][2]).unwrap_or(0.0);
    let dy = quad_peak_offset_1d(s[0][1], s[1][1], s[2][1]).unwrap_or(0.0);

    (center_x as f32 + dx, center_y as f32 + dy)
}

#[cfg(test)]
mod tests {
    use super::refine_subpixel_2d;

    #[test]
    fn refine_subpixel_separable_paraboloid() {
        let coords = [-1.0f32, 0.0, 1.0];
        let mut s = [[0.0f32; 3]; 3];
        for (yi, &y) in coords.iter().enumerate() {
            for (xi, &x) in coords.iter().enumerate() {
                s[yi][xi] = 1.0 - (x - 0.3).powi(2) - (y + 0.2).powi(2);
            }
        }

        let (x_ref, y_ref) = refine_subpixel_2d(0, 0, s);
        assert!((x_ref - 0.3).abs() < 1e-3);
        assert!((y_ref + 0.2).abs() < 1e-3);
    }
}
