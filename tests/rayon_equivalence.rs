#![cfg(feature = "rayon")]

use corrmatch::bank::{CompileConfig, CompiledTemplate};
use corrmatch::search::{MatchConfig, Matcher, Metric, RotationMode};
use corrmatch::template::rotate::rotate_u8_bilinear_masked;
use corrmatch::{ImageView, Template};

fn make_template(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let value = ((x * 11) ^ (y * 3) ^ (x * y)) & 0xFF;
            data.push(value as u8);
        }
    }
    data
}

#[test]
fn parallel_matches_sequential_rotation_enabled() {
    let tpl_width = 48;
    let tpl_height = 36;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let angle_deg = 30.0f32;
    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, angle_deg, 0);

    let img_width = 180;
    let img_height = 140;
    let x0 = 50;
    let y0 = 40;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            let idx = y * tpl_width + x;
            if mask[idx] == 1 {
                image[(y0 + y) * img_width + (x0 + x)] = rotated.data()[idx];
            }
        }
    }

    let compiled = CompiledTemplate::compile_rotated(
        &template,
        CompileConfig {
            max_levels: 2,
            coarse_step_deg: 30.0,
            min_step_deg: 15.0,
            fill_value: 0,
            precompute_coarsest: true,
        },
    )
    .unwrap();

    let base_cfg = MatchConfig {
        metric: Metric::Zncc,
        rotation: RotationMode::Enabled,
        max_image_levels: 2,
        beam_width: 6,
        per_angle_topk: 3,
        roi_radius: 6,
        nms_radius: 4,
        angle_half_range_steps: 1,
        ..MatchConfig::default()
    };

    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let seq_cfg = MatchConfig {
        parallel: false,
        ..base_cfg.clone()
    };
    let par_cfg = MatchConfig {
        parallel: true,
        ..base_cfg
    };
    let seq_matcher = Matcher::new(compiled).with_config(seq_cfg);
    let par_matcher = Matcher::new(
        CompiledTemplate::compile_rotated(
            &template,
            CompileConfig {
                max_levels: 2,
                coarse_step_deg: 30.0,
                min_step_deg: 15.0,
                fill_value: 0,
                precompute_coarsest: true,
            },
        )
        .unwrap(),
    )
    .with_config(par_cfg);

    let seq = seq_matcher.match_image(image_view).unwrap();
    let par = par_matcher.match_image(image_view).unwrap();

    let tol = 1e-6;
    assert!((seq.x - par.x).abs() <= tol);
    assert!((seq.y - par.y).abs() <= tol);
    assert!((seq.angle_deg - par.angle_deg).abs() <= tol);
    assert!((seq.score - par.score).abs() <= tol);
}
