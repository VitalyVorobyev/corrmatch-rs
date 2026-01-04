//! CorrMatch is a CPU-first template matching library focused on ZNCC.
//!
//! This crate currently provides scaffolding only; algorithms are not implemented yet.

pub mod bank;
mod candidate;
pub mod image;
mod kernel;
mod refine;
mod search;
pub mod template;
pub mod util;

pub use image::pyramid::ImagePyramid;
pub use image::ImageView;
pub use template::{Template, TemplatePlan};
pub use util::{CorrMatchError, CorrMatchResult};
