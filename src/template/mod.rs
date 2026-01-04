//! Template storage and planning utilities.

use crate::image::pyramid::OwnedImage;
use crate::image::ImageView;
use crate::util::CorrMatchResult;

mod plan;
pub(crate) mod rotate;

pub use plan::TemplatePlan;

/// Owned template image in contiguous grayscale format.
pub struct Template {
    img: OwnedImage,
}

impl Template {
    /// Creates a template from a contiguous grayscale buffer.
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> CorrMatchResult<Self> {
        let img = OwnedImage::from_vec(data, width, height)?;
        Ok(Self { img })
    }

    /// Returns a borrowed view of the template data.
    pub fn view(&self) -> ImageView<'_, u8> {
        self.img.view()
    }
}
