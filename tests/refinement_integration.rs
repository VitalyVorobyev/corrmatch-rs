use corrmatch::bank::{CompileConfig, CompileConfigNoRot, CompiledTemplate};
use corrmatch::search::{MatchConfig, Matcher, RotationMode};
use corrmatch::{ImageView, Template};

fn make_template(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let value = ((x * 7) ^ (y * 11) ^ (x * y)) & 0xFF;
            data.push(value as u8);
        }
    }
    data
}

fn make_mirror_template(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height);
    let mid = (width - 1) / 2;
    for y in 0..height {
        for x in 0..width {
            let sym_x = x.min(width - 1 - x).min(mid);
            let value = ((sym_x * 13 + y * 5 + sym_x * y) & 0xFF) as u8;
            data.push(value);
        }
    }
    data
}

fn angle_diff_deg(a: f32, b: f32) -> f32 {
    let mut diff = (a - b) % 360.0;
    if diff < -180.0 {
        diff += 360.0;
    }
    if diff >= 180.0 {
        diff -= 360.0;
    }
    diff.abs()
}

#[test]
fn refinement_handles_border_candidates() {
    let tpl_width = 12;
    let tpl_height = 10;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 40;
    let img_height = 30;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            image[y * img_width + x] = tpl_data[y * tpl_width + x];
        }
    }

    let compiled =
        CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot { max_levels: 1 })
            .unwrap();
    let cfg = MatchConfig {
        max_image_levels: 1,
        beam_width: 4,
        per_angle_topk: 2,
        roi_radius: 4,
        nms_radius: 3,
        angle_half_range_steps: 1,
        rotation: RotationMode::Disabled,
        ..MatchConfig::default()
    };
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let best = matcher.match_image(image_view).unwrap();

    assert!(best.x.is_finite());
    assert!(best.y.is_finite());
    assert!(best.angle_deg.is_finite());
    assert!(best.score.is_finite());
}

#[test]
fn refinement_keeps_center_angle_on_symmetric_template() {
    let tpl_width = 17;
    let tpl_height = 13;
    let tpl_data = make_mirror_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 60;
    let img_height = 50;
    let x0 = 14;
    let y0 = 12;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            image[(y0 + y) * img_width + (x0 + x)] = tpl_data[y * tpl_width + x];
        }
    }

    let compiled = CompiledTemplate::compile_rotated(
        &template,
        CompileConfig {
            max_levels: 1,
            coarse_step_deg: 30.0,
            min_step_deg: 30.0,
            fill_value: 0,
            precompute_coarsest: true,
        },
    )
    .unwrap();
    let step = compiled.angle_grid(0).unwrap().step_deg();
    let cfg = MatchConfig {
        max_image_levels: 1,
        beam_width: 4,
        per_angle_topk: 2,
        roi_radius: 4,
        nms_radius: 3,
        angle_half_range_steps: 1,
        rotation: RotationMode::Enabled,
        ..MatchConfig::default()
    };
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let best = matcher.match_image(image_view).unwrap();

    assert!(angle_diff_deg(best.angle_deg, 0.0) <= step * 0.25);
}
