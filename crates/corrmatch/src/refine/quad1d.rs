//! Quadratic 1D fitting for angle refinement.

/// Estimates the sub-sample peak offset for a quadratic fit.
///
/// Given samples at `x = -1, 0, +1` (`fm`, `f0`, `fp`), this returns the peak
/// offset `dx` in `[-1, 1]` when the fitted parabola is concave and stable.
///
/// The fitted parabola is assumed to be locally smooth and unimodal; if the
/// curvature is non-negative or ill-conditioned, `None` is returned.
pub fn quad_peak_offset_1d(fm: f32, f0: f32, fp: f32) -> Option<f32> {
    if !fm.is_finite() || !f0.is_finite() || !fp.is_finite() {
        return None;
    }

    let denom = fm - 2.0 * f0 + fp;
    let eps = 1e-6f32;
    if denom.abs() < eps || denom >= 0.0 {
        return None;
    }

    let dx = 0.5 * (fm - fp) / denom;
    if dx.is_finite() && dx.abs() <= 1.0 {
        Some(dx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::quad_peak_offset_1d;

    #[test]
    fn quad_peak_offset_symmetric() {
        let dx = quad_peak_offset_1d(0.9, 1.0, 0.9).unwrap();
        assert!(dx.abs() < 1e-6);
    }

    #[test]
    fn quad_peak_offset_shifted() {
        let f = |x: f32| 1.0 - (x - 0.25).powi(2);
        let fm = f(-1.0);
        let f0 = f(0.0);
        let fp = f(1.0);
        let dx = quad_peak_offset_1d(fm, f0, fp).unwrap();
        assert!((dx - 0.25).abs() < 1e-5);
    }

    #[test]
    fn quad_peak_offset_non_concave() {
        assert!(quad_peak_offset_1d(1.0, 0.5, 1.0).is_none());
    }
}
