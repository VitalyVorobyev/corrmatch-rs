//! CorrMatch is a CPU-first template matching library focused on ZNCC.
//!
//! This crate currently provides scaffolding only; algorithms are not implemented yet.

mod bank;
mod candidate;
mod image;
mod kernel;
mod refine;
mod search;
mod template;
mod util;

pub use util::error::{CorrMatchError, Result};
