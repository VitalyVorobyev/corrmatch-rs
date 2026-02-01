use crate::image::integral::IntegralImages;
use crate::kernel::Kernel;
use crate::kernel::{ScanParams, ScanRoi};
use crate::template::{MaskedSsdTemplatePlan, SsdTemplatePlan, TemplatePlan};
use crate::ImageView;
use std::collections::HashMap;

use super::SsdMaskedScalar;

#[cfg(not(feature = "simd"))]
use super::{SsdUnmaskedScalar as SsdUnmasked, ZnccUnmaskedScalar as ZnccUnmasked};
#[cfg(feature = "simd")]
use crate::kernel::simd::{SsdUnmaskedSimd as SsdUnmasked, ZnccUnmaskedSimd as ZnccUnmasked};

#[test]
fn unmasked_zncc_scan_matches_bruteforce() {
    let img_width = 6;
    let img_height = 5;
    let mut image = Vec::with_capacity(img_width * img_height);
    for y in 0..img_height {
        for x in 0..img_width {
            image.push(((x * 17 + y * 9 + x * y) & 0xFF) as u8);
        }
    }
    let tpl_width = 3;
    let tpl_height = 2;
    let mut tpl = Vec::with_capacity(tpl_width * tpl_height);
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            tpl.push(((x * 5 + y * 11 + x * y) & 0xFF) as u8);
        }
    }

    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
    let plan = TemplatePlan::from_view(tpl_view).unwrap();

    let params = ScanParams {
        topk: 1,
        min_var_i: 1e-8,
        min_score: f32::NEG_INFINITY,
    };
    let best = <ZnccUnmasked as Kernel>::scan_full(image_view, &plan, 0, params)
        .unwrap()
        .pop()
        .unwrap();

    let t_prime = plan.t_prime();
    let var_t = plan.var_t() as f64;
    let n = (tpl_width * tpl_height) as f64;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_x = 0;
    let mut best_y = 0;
    for y in 0..=(img_height - tpl_height) {
        for x in 0..=(img_width - tpl_width) {
            let mut dot = 0.0f64;
            let mut sum_i = 0.0f64;
            let mut sum_i2 = 0.0f64;
            for ty in 0..tpl_height {
                let row = image_view.row(y + ty).unwrap();
                let base = ty * tpl_width;
                for tx in 0..tpl_width {
                    let idx = base + tx;
                    let value = row[x + tx] as f64;
                    dot += t_prime[idx] as f64 * value;
                    sum_i += value;
                    sum_i2 += value * value;
                }
            }
            let var_i = sum_i2 - (sum_i * sum_i) / n;
            if var_i <= 1e-8 {
                continue;
            }
            let score = dot / (var_t * var_i).sqrt();
            if score > best_score {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    assert_eq!(best.x, best_x);
    assert_eq!(best.y, best_y);
    assert!((best.score - best_score as f32).abs() < 1e-5);
}

#[test]
fn unmasked_zncc_integral_scan_matches_scan_full() {
    let img_width = 6;
    let img_height = 5;
    let mut image = Vec::with_capacity(img_width * img_height);
    for y in 0..img_height {
        for x in 0..img_width {
            image.push(((x * 17 + y * 9 + x * y + 3) & 0xFF) as u8);
        }
    }
    let tpl_width = 3;
    let tpl_height = 2;
    let mut tpl = Vec::with_capacity(tpl_width * tpl_height);
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            tpl.push(((x * 5 + y * 11 + x * y + 1) & 0xFF) as u8);
        }
    }

    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
    let plan = TemplatePlan::from_view(tpl_view).unwrap();
    let placements = (img_width - tpl_width + 1) * (img_height - tpl_height + 1);
    let params = ScanParams {
        topk: placements,
        min_var_i: 1e-8,
        min_score: f32::NEG_INFINITY,
    };

    let scan_full = <ZnccUnmasked as Kernel>::scan_full(image_view, &plan, 0, params).unwrap();
    let integrals = IntegralImages::from_u8(image_view).unwrap();
    let integral =
        ZnccUnmasked::scan_full_integral(image_view, &plan, 0, params, &integrals).unwrap();

    assert_eq!(scan_full.len(), integral.len());
    let mut scores = HashMap::with_capacity(scan_full.len());
    for peak in scan_full {
        scores.insert((peak.x, peak.y), peak.score);
    }
    for peak in integral {
        let score = scores.get(&(peak.x, peak.y)).unwrap();
        assert!((peak.score - score).abs() < 1e-5);
    }

    // Basic smoke check for ROI variant (keeps clippy honest across feature combinations).
    let roi = ScanRoi::new(0, 0, 1, 1);
    let _roi_out =
        ZnccUnmasked::scan_roi_integral(image_view, &plan, 0, roi, params, &integrals).unwrap();
}

#[test]
fn unmasked_ssd_scan_matches_bruteforce() {
    let img_width = 5;
    let img_height = 4;
    let mut image = Vec::with_capacity(img_width * img_height);
    for y in 0..img_height {
        for x in 0..img_width {
            image.push(((x * 9 + y * 5 + x * y) & 0xFF) as u8);
        }
    }
    let tpl_width = 3;
    let tpl_height = 2;
    let mut tpl = Vec::with_capacity(tpl_width * tpl_height);
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            tpl.push(((x * 7 + y * 3 + x * y) & 0xFF) as u8);
        }
    }

    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
    let plan = SsdTemplatePlan::from_view(tpl_view).unwrap();

    let params = ScanParams {
        topk: 1,
        min_var_i: 0.0,
        min_score: f32::NEG_INFINITY,
    };
    let best = <SsdUnmasked as Kernel>::scan_full(image_view, &plan, 0, params)
        .unwrap()
        .pop()
        .unwrap();

    let data = plan.data();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_x = 0;
    let mut best_y = 0;
    for y in 0..=(img_height - tpl_height) {
        for x in 0..=(img_width - tpl_width) {
            let mut sse = 0.0f64;
            for ty in 0..tpl_height {
                let row = image_view.row(y + ty).unwrap();
                let base = ty * tpl_width;
                for tx in 0..tpl_width {
                    let idx = base + tx;
                    let value = row[x + tx] as f64;
                    let diff = value - data[idx] as f64;
                    sse += diff * diff;
                }
            }
            let score = -sse;
            if score > best_score {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    assert_eq!(best.x, best_x);
    assert_eq!(best.y, best_y);
    assert!((best.score - best_score as f32).abs() < 1e-5);
}

#[test]
fn masked_ssd_score_at_matches_expected() {
    let tpl_width = 3;
    let tpl_height = 3;
    let tpl = vec![
        1u8, 2, 3, //
        4, 5, 6, //
        7, 8, 9,
    ];
    let mask = vec![
        0u8, 0, 0, //
        0, 1, 0, //
        0, 0, 0,
    ];
    let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height).unwrap();
    let plan = MaskedSsdTemplatePlan::from_rotated_u8(tpl_view, mask, 0.0).unwrap();

    let image = vec![
        0u8, 0, 0, 0, //
        0, 10, 0, 0, //
        0, 0, 0, 0, //
        0, 0, 0, 0,
    ];
    let image_view = ImageView::from_slice(&image, 4, 4).unwrap();
    let score = <SsdMaskedScalar as Kernel>::score_at(image_view, &plan, 0, 0, 0.0);

    let diff = 10.0f32 - 5.0f32;
    let expected = -(diff * diff);
    assert!((score - expected).abs() < 1e-6);
}
