//! CorrMatch is a CPU-first template matching library focused on ZNCC.
//!
//! This crate currently provides scaffolding only; algorithms are not implemented yet.

pub mod bank;
mod candidate;
pub mod image;
mod kernel;
mod refine;
pub mod search;
pub mod template;
pub mod util;

pub use image::pyramid::ImagePyramid;
pub use image::ImageView;
pub use template::{MaskedTemplatePlan, Template, TemplatePlan};
pub use util::{CorrMatchError, CorrMatchResult};

pub use candidate::nms::nms_2d;
pub use candidate::topk::{Peak, TopK};
pub use search::scan::{
    scan_masked_zncc_scalar, scan_masked_zncc_scalar_full, scan_masked_zncc_scalar_roi,
};
pub use search::{Match, MatchConfig, Matcher};
