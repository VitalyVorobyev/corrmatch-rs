//! Template storage and planning utilities.

use crate::image::{ImageView, OwnedImage};
use crate::util::CorrMatchResult;

mod plan;
pub mod rotate;

pub use plan::{MaskedTemplatePlan, TemplatePlan};

/// Owned template image in contiguous grayscale format.
pub struct Template {
    img: OwnedImage,
}

impl Template {
    /// Creates a template from a contiguous grayscale buffer.
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> CorrMatchResult<Self> {
        let img = OwnedImage::new(data, width, height)?;
        Ok(Self { img })
    }

    /// Returns a borrowed view of the template data.
    pub fn view(&self) -> ImageView<'_, u8> {
        self.img.view()
    }
}
