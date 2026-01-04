//! Low-level building blocks for custom matching pipelines.
//!
//! These types expose template plans, kernel traits, and scan helpers for
//! advanced use cases beyond the high-level `Matcher` API.

pub use crate::candidate::nms::nms_2d;
pub use crate::candidate::topk::{Peak, TopK};
pub use crate::kernel::{Kernel, ScanParams};
pub use crate::search::scan::{
    scan_masked_zncc_scalar, scan_masked_zncc_scalar_full, scan_masked_zncc_scalar_roi,
    score_masked_zncc_at,
};
pub use crate::template::{
    MaskedSsdTemplatePlan, MaskedTemplatePlan, SsdTemplatePlan, TemplatePlan,
};
