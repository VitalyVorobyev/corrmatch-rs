//! CorrMatch is a CPU-first template matching library focused on ZNCC.
//!
//! This crate currently provides scaffolding only; algorithms are not implemented yet.

pub mod image;
pub mod template;
pub mod util;

mod bank;
mod candidate;
mod kernel;
mod refine;
mod search;

pub use image::pyramid::ImagePyramid;
pub use image::ImageView;
pub use template::{Template, TemplatePlan};
pub use util::{CorrMatchError, CorrMatchResult};
