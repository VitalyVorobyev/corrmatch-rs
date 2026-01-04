//! Error types for corrmatch.

use thiserror::Error;

/// Result alias for corrmatch operations.
pub type Result<T> = std::result::Result<T, CorrMatchError>;

/// Errors that can occur when running corrmatch algorithms.
#[derive(Debug, Error)]
pub enum CorrMatchError {
    /// The input data or parameters are invalid.
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    /// Placeholder for unfinished features.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}
