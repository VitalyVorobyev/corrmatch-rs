//! Mathematical helpers for correlation and refinement.

/// Wraps an angle in degrees to the range [-180, 180).
pub(crate) fn wrap_deg(angle_deg: f32) -> f32 {
    let mut wrapped = angle_deg % 360.0;
    if wrapped < -180.0 {
        wrapped += 360.0;
    }
    if wrapped >= 180.0 {
        wrapped -= 360.0;
    }
    wrapped
}

/// Converts degrees to radians.
pub(crate) fn deg_to_rad(angle_deg: f32) -> f32 {
    angle_deg.to_radians()
}

/// Computes sine and cosine for an angle in degrees.
pub(crate) fn sin_cos_deg(angle_deg: f32) -> (f32, f32) {
    let radians = deg_to_rad(angle_deg);
    radians.sin_cos()
}

#[cfg(test)]
mod tests {
    use super::{deg_to_rad, sin_cos_deg, wrap_deg};

    #[test]
    fn wrap_deg_maps_to_expected_range() {
        assert!((wrap_deg(181.0) + 179.0).abs() < 1e-6);
        assert!((wrap_deg(-181.0) - 179.0).abs() < 1e-6);
        assert!((wrap_deg(540.0) + 180.0).abs() < 1e-6);
    }

    #[test]
    fn deg_to_rad_matches_pi() {
        let radians = deg_to_rad(180.0);
        assert!((radians - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn sin_cos_deg_matches_quadrants() {
        let (sin, cos) = sin_cos_deg(90.0);
        assert!(sin > 0.999);
        assert!(cos.abs() < 1e-6);
    }
}
