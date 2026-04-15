//! Low-level building blocks for custom matching pipelines.
//!
//! These types expose template plans, kernel traits, and scan helpers for
//! advanced use cases beyond the high-level `Matcher` API. Most users should
//! prefer the top-level [`Template`](crate::Template), [`CompiledTemplate`](crate::CompiledTemplate),
//! and [`Matcher`](crate::Matcher) types.
//!
//! # Example: single-level masked ZNCC scan
//!
//! The example below shows the minimal call sequence for running a dense masked
//! ZNCC scan at a single pyramid level without going through `Matcher`:
//!
//! 1. Build a [`MaskedTemplatePlan`] from a template image and binary mask.
//! 2. Construct a target [`ImageView`](crate::ImageView).
//! 3. Call [`scan_masked_zncc_scalar_full`] and inspect the [`Peak`] results.
//!
//! ```no_run
//! use corrmatch::lowlevel::{
//!     MaskedTemplatePlan, scan_masked_zncc_scalar_full,
//! };
//! use corrmatch::ImageView;
//!
//! // --- Build a tiny 4×4 template ---
//! // In a real application this would come from an image file.
//! let tpl_width = 4usize;
//! let tpl_height = 4usize;
//! let tpl_pixels: Vec<u8> = (0u8..16).collect();
//! let tpl_view = ImageView::new(&tpl_pixels, tpl_width, tpl_height, tpl_width)
//!     .expect("valid template dimensions");
//!
//! // All pixels are unmasked (255 = valid).
//! let mask: Vec<u8> = vec![255u8; tpl_width * tpl_height];
//!
//! // Build the precomputed plan (computes masked mean, variance, zero-mean buffer).
//! let plan = MaskedTemplatePlan::from_rotated_u8(tpl_view, mask, 0.0)
//!     .expect("non-degenerate template");
//!
//! // --- Scan a 16×16 search image ---
//! let img_width = 16usize;
//! let img_height = 16usize;
//! let img_pixels: Vec<u8> = (0u8..=255).cycle().take(img_width * img_height).collect();
//! let img_view = ImageView::new(&img_pixels, img_width, img_height, img_width)
//!     .expect("valid image dimensions");
//!
//! // angle_idx = 0 (no rotation bank), topk = 3, min_var_i = 1e-8, min_score = -∞
//! let peaks = scan_masked_zncc_scalar_full(img_view, &plan, 0, 3, 1e-8, f32::NEG_INFINITY)
//!     .expect("scan succeeded");
//!
//! for peak in &peaks {
//!     println!(
//!         "x={} y={} angle_idx={} score={:.4}",
//!         peak.x, peak.y, peak.angle_idx, peak.score
//!     );
//! }
//! ```

pub use crate::bank::AngleGrid;
pub use crate::candidate::nms::nms_2d;
pub use crate::candidate::topk::{Peak, TopK};
pub use crate::kernel::{Kernel, ScanParams};
pub use crate::search::scan::{
    scan_masked_zncc_scalar, scan_masked_zncc_scalar_full, scan_masked_zncc_scalar_roi,
    score_masked_zncc_at,
};
pub use crate::template::rotate::{rotate_u8_bilinear, rotate_u8_bilinear_masked};
pub use crate::template::{
    MaskedSsdTemplatePlan, MaskedTemplatePlan, SsdTemplatePlan, TemplatePlan, ValidCoord,
};
