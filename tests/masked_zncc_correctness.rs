use corrmatch::lowlevel::{nms_2d, scan_masked_zncc_scalar, MaskedTemplatePlan, Peak};
use corrmatch::template::rotate::rotate_u8_bilinear_masked;
use corrmatch::ImageView;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn brute_force_best(image: ImageView<'_, u8>, tpl: &MaskedTemplatePlan) -> (usize, usize, f64) {
    let tpl_w = tpl.width();
    let tpl_h = tpl.height();
    let sum_w = tpl.sum_w() as f64;
    let var_t = tpl.var_t() as f64;
    let t_prime = tpl.t_prime();
    let mask = tpl.mask();

    let max_y = image.height() - tpl_h;
    let max_x = image.width() - tpl_w;

    let mut best_x = 0usize;
    let mut best_y = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for y in 0..=max_y {
        for x in 0..=max_x {
            let mut dot = 0.0f64;
            let mut sum_i = 0.0f64;
            let mut sum_i2 = 0.0f64;

            for ty in 0..tpl_h {
                let row = image.row(y + ty).expect("row in bounds");
                let base = ty * tpl_w;
                for tx in 0..tpl_w {
                    let idx = base + tx;
                    let w = if mask[idx] == 0 { 0.0 } else { 1.0 };
                    let value = row[x + tx] as f64;
                    dot += t_prime[idx] as f64 * value;
                    sum_i += w * value;
                    sum_i2 += w * value * value;
                }
            }

            let var_i = sum_i2 - (sum_i * sum_i) / sum_w;
            if var_i <= 1e-12 {
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

    (best_x, best_y, best_score)
}

#[test]
fn zncc_finds_perfect_match_no_rotation() {
    let mut rng = StdRng::seed_from_u64(123);
    let width = 32;
    let height = 32;
    let mut image = vec![0u8; width * height];
    for value in image.iter_mut() {
        *value = rng.random_range(0..=255);
    }

    let tpl_width = 11;
    let tpl_height = 9;
    let x0 = 7;
    let y0 = 9;
    let mut tpl_data = Vec::with_capacity(tpl_width * tpl_height);
    for y in 0..tpl_height {
        let row = &image[(y0 + y) * width..(y0 + y) * width + width];
        tpl_data.extend_from_slice(&row[x0..x0 + tpl_width]);
    }

    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, 0.0, 0);
    let plan = MaskedTemplatePlan::from_rotated_u8(rotated.view(), mask, 0.0).unwrap();

    let image_view = ImageView::from_slice(&image, width, height).unwrap();
    let peaks = scan_masked_zncc_scalar(image_view, &plan, 0, 5).unwrap();
    let best = peaks.first().expect("at least one peak");
    assert_eq!(best.x, x0);
    assert_eq!(best.y, y0);
    assert!(best.score > 0.99);
}

#[test]
fn zncc_finds_rotated_match() {
    let tpl_width = 17;
    let tpl_height = 17;
    let mut tpl_data = Vec::with_capacity(tpl_width * tpl_height);
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            tpl_data.push(((x * 17 + y * 31) % 255) as u8);
        }
    }

    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();
    let angle = 30.0f32;
    let fill = 0u8;
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, angle, fill);
    let plan = MaskedTemplatePlan::from_rotated_u8(rotated.view(), mask.clone(), angle).unwrap();

    let width = 40;
    let height = 40;
    let x0 = 10;
    let y0 = 8;
    let mut image = vec![fill; width * height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            let idx = y * tpl_width + x;
            if mask[idx] == 1 {
                let dst_idx = (y0 + y) * width + (x0 + x);
                image[dst_idx] = rotated.data()[idx];
            }
        }
    }

    let image_view = ImageView::from_slice(&image, width, height).unwrap();
    let peaks = scan_masked_zncc_scalar(image_view, &plan, 0, 3).unwrap();
    let best = peaks.first().expect("at least one peak");
    assert_eq!(best.x, x0);
    assert_eq!(best.y, y0);
    assert!(best.score > 0.99);
}

#[test]
fn zncc_matches_bruteforce_on_small_case() {
    let width = 10;
    let height = 10;
    let mut image = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            image[y * width + x] = (x + y * width) as u8;
        }
    }

    let tpl_width = 4;
    let tpl_height = 3;
    let x0 = 2;
    let y0 = 4;
    let mut tpl_data = Vec::with_capacity(tpl_width * tpl_height);
    for y in 0..tpl_height {
        let row = &image[(y0 + y) * width..(y0 + y) * width + width];
        tpl_data.extend_from_slice(&row[x0..x0 + tpl_width]);
    }

    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, 0.0, 0);
    let plan = MaskedTemplatePlan::from_rotated_u8(rotated.view(), mask, 0.0).unwrap();

    let image_view = ImageView::from_slice(&image, width, height).unwrap();
    let (bx, by, bscore) = brute_force_best(image_view, &plan);
    let peaks = scan_masked_zncc_scalar(image_view, &plan, 0, 1).unwrap();
    let best = peaks.first().expect("at least one peak");
    assert_eq!(best.x, bx);
    assert_eq!(best.y, by);
    assert!((best.score as f64 - bscore).abs() < 1e-5);
}

#[test]
fn nms_reduces_nearby_peaks() {
    let mut peaks = vec![
        Peak {
            x: 10,
            y: 10,
            score: 0.9,
            angle_idx: 0,
        },
        Peak {
            x: 11,
            y: 10,
            score: 0.8,
            angle_idx: 0,
        },
        Peak {
            x: 20,
            y: 20,
            score: 0.7,
            angle_idx: 0,
        },
    ];

    let kept = nms_2d(&mut peaks, 1);
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].x, 10);
    assert_eq!(kept[0].y, 10);
    assert_eq!(kept[1].x, 20);
    assert_eq!(kept[1].y, 20);
}

#[test]
fn nms_keeps_all_with_zero_radius() {
    let mut peaks = vec![
        Peak {
            x: 1,
            y: 1,
            score: 0.2,
            angle_idx: 0,
        },
        Peak {
            x: 2,
            y: 2,
            score: 0.5,
            angle_idx: 0,
        },
    ];

    let kept = nms_2d(&mut peaks, 0);
    assert_eq!(kept.len(), 2);
    assert!(kept[0].score >= kept[1].score);
}
