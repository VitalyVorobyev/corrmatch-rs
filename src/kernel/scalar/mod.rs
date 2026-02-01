//! Scalar reference kernels for score evaluation.

mod common;
mod masked_ssd;
mod masked_zncc;

// Unmasked scalar kernels are used only when SIMD is disabled.
#[cfg(not(feature = "simd"))]
mod unmasked_ssd;
#[cfg(not(feature = "simd"))]
mod unmasked_zncc;

pub use masked_ssd::SsdMaskedScalar;
pub use masked_zncc::ZnccMaskedScalar;

#[cfg(not(feature = "simd"))]
pub use unmasked_ssd::SsdUnmaskedScalar;
#[cfg(not(feature = "simd"))]
pub use unmasked_zncc::ZnccUnmaskedScalar;

#[cfg(test)]
mod tests;
