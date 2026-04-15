use crate::kernel::ScanRoi;
use crate::util::{CorrMatchError, CorrMatchResult};
use crate::ImageView;

pub(crate) fn clamp_scan_roi(
    image: ImageView<'_, u8>,
    tpl_width: usize,
    tpl_height: usize,
    roi: ScanRoi,
) -> CorrMatchResult<Option<ScanRoi>> {
    let img_width = image.width();
    let img_height = image.height();
    if img_width < tpl_width || img_height < tpl_height {
        return Err(CorrMatchError::RoiOutOfBounds {
            x: 0,
            y: 0,
            width: tpl_width,
            height: tpl_height,
            img_width,
            img_height,
        });
    }

    let max_x = img_width - tpl_width;
    let max_y = img_height - tpl_height;
    Ok(roi.clamp_inclusive(max_x, max_y))
}
