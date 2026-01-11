//! Template storage and planning utilities.

use crate::bank::{CompileConfig, CompiledTemplate};
use crate::image::{ImageView, OwnedImage};
use crate::util::CorrMatchResult;

mod plan;
pub mod rotate;

pub use plan::{MaskedSsdTemplatePlan, MaskedTemplatePlan, SsdTemplatePlan, TemplatePlan};

/// Owned template image in contiguous grayscale format.
///
/// Use `Template::compile` to build reusable assets for matching.
pub struct Template {
    img: OwnedImage,
}

impl Template {
    /// Creates a template from a contiguous grayscale buffer.
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> CorrMatchResult<Self> {
        let img = OwnedImage::new(data, width, height)?;
        Ok(Self { img })
    }

    /// Returns the template width.
    pub fn width(&self) -> usize {
        self.img.width()
    }

    /// Returns the template height.
    pub fn height(&self) -> usize {
        self.img.height()
    }

    /// Returns a borrowed view of the template data.
    pub fn view(&self) -> ImageView<'_, u8> {
        self.img.view()
    }

    /// Compiles template assets for matching with rotation support.
    ///
    /// For translation-only matching, use `CompiledTemplate::compile_unrotated`.
    pub fn compile(&self, cfg: CompileConfig) -> CorrMatchResult<CompiledTemplate> {
        CompiledTemplate::compile_rotated(self, cfg)
    }
}
