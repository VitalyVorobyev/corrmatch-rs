//! CorrMatch is a CPU-first template matching library focused on ZNCC/SSD.
//!
//! This crate provides baseline scalar kernels and a coarse-to-fine matcher,
//! with optional parallelism via the `rayon` feature; SIMD acceleration is
//! planned.

pub mod bank;
mod candidate;
pub mod image;
pub mod kernel;
mod refine;
pub mod search;
pub mod template;
pub mod util;

pub use image::pyramid::ImagePyramid;
pub use image::ImageView;
pub use kernel::{Kernel, ScanParams};
pub use template::{
    MaskedSsdTemplatePlan, MaskedTemplatePlan, SsdTemplatePlan, Template, TemplatePlan,
};
pub use util::{CorrMatchError, CorrMatchResult};

pub use candidate::nms::nms_2d;
pub use candidate::topk::{Peak, TopK};
pub use search::scan::{
    scan_masked_zncc_scalar, scan_masked_zncc_scalar_full, scan_masked_zncc_scalar_roi,
    score_masked_zncc_at,
};
pub use search::{Match, MatchConfig, Matcher, Metric, RotationMode};
