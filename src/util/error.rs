//! Error types for corrmatch.

use thiserror::Error;

/// Result alias for corrmatch operations.
pub type CorrMatchResult<T> = std::result::Result<T, CorrMatchError>;

/// Errors that can occur when running corrmatch operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CorrMatchError {
    /// The provided dimensions are invalid (must be non-zero).
    #[error("invalid dimensions: width={width} height={height}")]
    InvalidDimensions { width: usize, height: usize },
    /// The provided stride is smaller than the image width.
    #[error("invalid stride: width={width} stride={stride}")]
    InvalidStride { width: usize, stride: usize },
    /// The backing buffer is too small for the requested view.
    #[error("buffer too small: needed={needed} got={got}")]
    BufferTooSmall { needed: usize, got: usize },
    /// The requested ROI lies outside the image bounds.
    #[error(
        "roi out of bounds: x={x} y={y} width={width} height={height} img_width={img_width} img_height={img_height}"
    )]
    RoiOutOfBounds {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        img_width: usize,
        img_height: usize,
    },
    /// The template is degenerate and cannot be normalized.
    #[error("degenerate template: {reason}")]
    DegenerateTemplate { reason: &'static str },
    /// The requested angle grid parameters are invalid.
    #[error("invalid angle grid: {reason}")]
    InvalidAngleGrid { reason: &'static str },
    /// The requested index is out of bounds for a collection.
    #[error("index out of bounds: {context} index={index} len={len}")]
    IndexOutOfBounds {
        index: usize,
        len: usize,
        context: &'static str,
    },
}
