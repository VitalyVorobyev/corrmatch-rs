use corrmatch::lowlevel::rotate_u8_bilinear_masked;
use corrmatch::{
    CompileConfig, CompileConfigNoRot, CompiledTemplate, ImageView, MatchConfig, Matcher, Metric,
    RotationMode, Template,
};

fn make_template(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let value = ((x * 13) ^ (y * 7) ^ (x * y)) & 0xFF;
            data.push(value as u8);
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
fn pipeline_finds_rotated_match() {
    let tpl_width = 64;
    let tpl_height = 48;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let angle_deg = 30.0f32;
    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, angle_deg, 0);

    let img_width = 220;
    let img_height = 180;
    let x0 = 70;
    let y0 = 50;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            let idx = y * tpl_width + x;
            if mask[idx] == 1 {
                image[(y0 + y) * img_width + (x0 + x)] = rotated.data()[idx];
            }
        }
    }

    // Use a single level so rotation and downsampling do not introduce mismatch.
    let mut compile_cfg = CompileConfig::default();
    compile_cfg.max_levels = 1;
    compile_cfg.coarse_step_deg = 30.0;
    compile_cfg.min_step_deg = 30.0;
    compile_cfg.fill_value = 0;
    compile_cfg.precompute_coarsest = true;
    let compiled = CompiledTemplate::compile_rotated(&template, compile_cfg).unwrap();
    let level0_step = compiled.angle_grid(0).unwrap().step_deg();

    let mut cfg = MatchConfig::default();
    cfg.max_image_levels = 1;
    cfg.beam_width = 8;
    cfg.per_angle_topk = 3;
    cfg.roi_radius = 8;
    cfg.nms_radius = 6;
    cfg.angle_half_range_steps = 1;
    cfg.rotation = RotationMode::Enabled;
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let best = matcher.match_image(image_view).unwrap();

    assert!(
        (best.x - x0 as f32).abs() <= 4.0,
        "expected x near {x0}, got {}",
        best.x
    );
    assert!(
        (best.y - y0 as f32).abs() <= 4.0,
        "expected y near {y0}, got {}",
        best.y
    );

    assert!(
        angle_diff_deg(best.angle_deg, angle_deg) <= level0_step + 1e-6,
        "expected angle near {angle_deg}, got {}",
        best.angle_deg
    );
    assert!(best.score > 0.95);
}

#[test]
fn pipeline_finds_translation_match() {
    let tpl_width = 40;
    let tpl_height = 32;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 160;
    let img_height = 120;
    let x0 = 33;
    let y0 = 41;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            image[(y0 + y) * img_width + (x0 + x)] = tpl_data[y * tpl_width + x];
        }
    }

    let mut compile_cfg = CompileConfig::default();
    compile_cfg.max_levels = 4;
    compile_cfg.coarse_step_deg = 45.0;
    compile_cfg.min_step_deg = 45.0;
    compile_cfg.fill_value = 0;
    compile_cfg.precompute_coarsest = true;
    let compiled = CompiledTemplate::compile_rotated(&template, compile_cfg).unwrap();
    let step = compiled.angle_grid(0).unwrap().step_deg();

    let mut cfg = MatchConfig::default();
    cfg.beam_width = 6;
    cfg.per_angle_topk = 3;
    cfg.roi_radius = 6;
    cfg.nms_radius = 4;
    cfg.angle_half_range_steps = 1;
    cfg.rotation = RotationMode::Enabled;
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let best = matcher.match_image(image_view).unwrap();

    assert!((best.x - x0 as f32).abs() <= 1.0);
    assert!((best.y - y0 as f32).abs() <= 1.0);
    assert!(angle_diff_deg(best.angle_deg, 0.0) <= step + 1e-6);
    assert!(best.score > 0.99);
}

#[test]
fn angle_step_schedule_matches_expected() {
    let tpl_width = 16;
    let tpl_height = 16;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data, tpl_width, tpl_height).unwrap();

    let mut compile_cfg = CompileConfig::default();
    compile_cfg.max_levels = 3;
    compile_cfg.coarse_step_deg = 20.0;
    compile_cfg.min_step_deg = 1.0;
    compile_cfg.fill_value = 0;
    compile_cfg.precompute_coarsest = false;
    let compiled = CompiledTemplate::compile_rotated(&template, compile_cfg).unwrap();

    assert_eq!(compiled.num_levels(), 3);
    let step0 = compiled.angle_grid(0).unwrap().step_deg();
    let step1 = compiled.angle_grid(1).unwrap().step_deg();
    let step2 = compiled.angle_grid(2).unwrap().step_deg();

    assert!((step2 - 20.0).abs() < 1e-6);
    assert!((step1 - 10.0).abs() < 1e-6);
    assert!((step0 - 5.0).abs() < 1e-6);
}

#[test]
fn pipeline_finds_translation_match_rotation_disabled() {
    let tpl_width = 32;
    let tpl_height = 24;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 140;
    let img_height = 110;
    let x0 = 21;
    let y0 = 17;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            image[(y0 + y) * img_width + (x0 + x)] = tpl_data[y * tpl_width + x];
        }
    }

    let mut norot_cfg = CompileConfigNoRot::default();
    norot_cfg.max_levels = 3;
    let compiled = CompiledTemplate::compile_unrotated(&template, norot_cfg).unwrap();

    let mut cfg = MatchConfig::default();
    cfg.max_image_levels = 3;
    cfg.beam_width = 5;
    cfg.per_angle_topk = 3;
    cfg.roi_radius = 6;
    cfg.nms_radius = 4;
    cfg.rotation = RotationMode::Disabled;
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let best = matcher.match_image(image_view).unwrap();

    assert!((best.x - x0 as f32).abs() <= 1.0);
    assert!((best.y - y0 as f32).abs() <= 1.0);
    assert!(angle_diff_deg(best.angle_deg, 0.0) <= 1e-6);
    assert!(best.score > 0.99);
}

#[test]
fn pipeline_finds_translation_match_ssd_rotation_disabled() {
    let tpl_width = 24;
    let tpl_height = 20;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 120;
    let img_height = 90;
    let x0 = 17;
    let y0 = 21;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            image[(y0 + y) * img_width + (x0 + x)] = tpl_data[y * tpl_width + x];
        }
    }

    let mut norot_cfg = CompileConfigNoRot::default();
    norot_cfg.max_levels = 3;
    let compiled = CompiledTemplate::compile_unrotated(&template, norot_cfg).unwrap();

    let mut cfg = MatchConfig::default();
    cfg.metric = Metric::Ssd;
    cfg.rotation = RotationMode::Disabled;
    cfg.max_image_levels = 3;
    cfg.beam_width = 5;
    cfg.per_angle_topk = 3;
    cfg.roi_radius = 6;
    cfg.nms_radius = 4;
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();
    let best = matcher.match_image(image_view).unwrap();

    assert!((best.x - x0 as f32).abs() <= 1.0);
    assert!((best.y - y0 as f32).abs() <= 1.0);
    assert!(best.score >= -1e-6);
}

#[test]
fn pipeline_topk_returns_best_first() {
    let tpl_width = 24;
    let tpl_height = 18;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 120;
    let img_height = 90;
    let x0 = 19;
    let y0 = 23;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            image[(y0 + y) * img_width + (x0 + x)] = tpl_data[y * tpl_width + x];
        }
    }

    let mut norot_cfg = CompileConfigNoRot::default();
    norot_cfg.max_levels = 3;
    let compiled = CompiledTemplate::compile_unrotated(&template, norot_cfg).unwrap();
    let mut cfg = MatchConfig::default();
    cfg.max_image_levels = 3;
    cfg.rotation = RotationMode::Disabled;
    let matcher = Matcher::new(compiled).with_config(cfg);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();

    let best = matcher.match_image(image_view).unwrap();
    let topk = matcher.match_image_topk(image_view, 3).unwrap();

    assert!(!topk.is_empty());
    assert!((topk[0].x - best.x).abs() < 1e-6);
    assert!((topk[0].y - best.y).abs() < 1e-6);
    assert!((topk[0].angle_deg - best.angle_deg).abs() < 1e-6);
    assert!((topk[0].score - best.score).abs() < 1e-6);
}
