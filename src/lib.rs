//! CorrMatch is a CPU-first template matching library for grayscale images.
//!
//! It provides a coarse-to-fine matcher with optional rotation search and
//! two metrics: ZNCC and SSD. The primary entry points are `Template`,
//! `CompiledTemplate`, and `Matcher`.
//!
//! # Quick start
//! ```no_run
//! use corrmatch::{
//!     CompileConfig, MatchConfig, Matcher, RotationMode, Template, ImageView,
//! };
//!
//! # fn run(image: &[u8], width: usize, height: usize, tpl: Vec<u8>, tw: usize, th: usize)
//! #     -> corrmatch::CorrMatchResult<corrmatch::Match> {
//! let template = Template::new(tpl, tw, th)?;
//! let compiled = template.compile(CompileConfig::default())?;
//! let matcher = Matcher::new(compiled).with_config(MatchConfig {
//!     rotation: RotationMode::Enabled,
//!     ..MatchConfig::default()
//! });
//! let image_view = ImageView::from_slice(image, width, height)?;
//! matcher.match_image(image_view)
//! # }
//! ```
//!
//! # Concepts
//! - `Template`: owned template pixels.
//! - `CompiledTemplate`: precomputed template pyramid and (optionally) angle banks.
//! - `Matcher`: runs the coarse-to-fine search and returns the best match.
//!
//! # Feature flags
//! - `rayon`: parallel search execution.
//! - `simd`: SIMD-accelerated kernels (planned).
//! - `image-io`: file I/O helpers via the `image` crate.
//!
//! # Low-level API
//! Advanced building blocks are available under `corrmatch::lowlevel`.
//!
//! # CLI
//! A JSON-driven CLI lives in the `corrmatch-cli` workspace crate.

mod bank;
mod candidate;
mod image;
mod kernel;
pub mod lowlevel;
mod refine;
mod search;
mod template;
mod util;

pub use bank::{CompileConfig, CompileConfigNoRot, CompiledTemplate};
pub use image::pyramid::ImagePyramid;
pub use image::{ImageView, OwnedImage};
pub use template::Template;
pub use util::{CorrMatchError, CorrMatchResult};

pub use search::{Match, MatchConfig, Matcher, Metric, RotationMode};

/// Image I/O helpers available when the `image-io` feature is enabled.
#[cfg(feature = "image-io")]
pub mod io {
    pub use crate::image::io::{
        load_gray_image, owned_from_dynamic_image, owned_from_gray_image, view_from_gray_image,
    };
}
