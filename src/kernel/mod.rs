//! Correlation kernel implementations.

pub(crate) mod scalar;

#[cfg(feature = "simd")]
pub(crate) mod simd;

#[cfg(feature = "rayon")]
pub(crate) mod rayon;
