//! CorrMatch is a CPU-first template matching library focused on ZNCC/SSD.
//!
//! This crate provides baseline scalar kernels and a coarse-to-fine matcher,
//! with optional parallelism via the `rayon` feature; SIMD acceleration is
//! planned.

pub mod bank;
mod candidate;
pub mod image;
mod kernel;
pub mod lowlevel;
mod refine;
pub mod search;
pub mod template;
pub mod util;

pub use bank::{CompileConfig, CompileConfigNoRot, CompiledTemplate};
pub use image::pyramid::ImagePyramid;
pub use image::ImageView;
pub use template::Template;
pub use util::{CorrMatchError, CorrMatchResult};

pub use search::{Match, MatchConfig, Matcher, Metric, RotationMode};
