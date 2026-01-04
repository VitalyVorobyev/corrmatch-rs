use corrmatch::lowlevel::rotate_u8_bilinear_masked;
use corrmatch::{
    CompileConfig, CompileConfigNoRot, CompiledTemplate, ImageView, MatchConfig, Matcher, Metric,
    RotationMode, Template,
};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn make_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let value = ((x * 13) ^ (y * 7) ^ (x * y)) & 0xFF;
            data.push(value as u8);
        }
    }
    data
}

fn extract_patch(
    image: &[u8],
    img_width: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let row = (y0 + y) * img_width;
        for x in 0..width {
            out.push(image[row + x0 + x]);
        }
    }
    out
}

fn bench_matcher(c: &mut Criterion) {
    let img_width = 512;
    let img_height = 512;
    let image = make_image(img_width, img_height);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();

    let tpl_width = 192;
    let tpl_height = 192;
    let tpl_x0 = 120;
    let tpl_y0 = 100;
    let tpl_data = extract_patch(&image, img_width, tpl_x0, tpl_y0, tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let compiled_unmasked =
        CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot { max_levels: 4 })
            .unwrap();
    let matcher_unmasked = Matcher::new(compiled_unmasked).with_config(MatchConfig {
        metric: Metric::Zncc,
        rotation: RotationMode::Disabled,
        parallel: false,
        max_image_levels: 4,
        beam_width: 6,
        per_angle_topk: 3,
        roi_radius: 6,
        nms_radius: 4,
        angle_half_range_steps: 1,
        ..MatchConfig::default()
    });

    c.bench_function("zncc_unmasked_rotation_off", |b| {
        b.iter(|| black_box(matcher_unmasked.match_image(image_view).unwrap()));
    });

    let matcher_ssd_unmasked = Matcher::new(
        CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot { max_levels: 4 })
            .unwrap(),
    )
    .with_config(MatchConfig {
        metric: Metric::Ssd,
        rotation: RotationMode::Disabled,
        parallel: false,
        max_image_levels: 4,
        beam_width: 6,
        per_angle_topk: 3,
        roi_radius: 6,
        nms_radius: 4,
        angle_half_range_steps: 1,
        ..MatchConfig::default()
    });

    c.bench_function("ssd_unmasked_rotation_off", |b| {
        b.iter(|| black_box(matcher_ssd_unmasked.match_image(image_view).unwrap()));
    });

    if cfg!(feature = "rayon") {
        let matcher_unmasked_par = Matcher::new(
            CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot { max_levels: 4 })
                .unwrap(),
        )
        .with_config(MatchConfig {
            metric: Metric::Zncc,
            rotation: RotationMode::Disabled,
            parallel: true,
            max_image_levels: 4,
            beam_width: 6,
            per_angle_topk: 3,
            roi_radius: 6,
            nms_radius: 4,
            angle_half_range_steps: 1,
            ..MatchConfig::default()
        });

        c.bench_function("zncc_unmasked_rotation_off_parallel", |b| {
            b.iter(|| black_box(matcher_unmasked_par.match_image(image_view).unwrap()));
        });
    }

    let rotated_angle = 30.0f32;
    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, rotated_angle, 0);
    let mut image_rot = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        for x in 0..tpl_width {
            let idx = y * tpl_width + x;
            if mask[idx] == 1 {
                image_rot[(tpl_y0 + y) * img_width + (tpl_x0 + x)] = rotated.data()[idx];
            }
        }
    }
    let image_rot_view = ImageView::from_slice(&image_rot, img_width, img_height).unwrap();

    let compiled_rot = CompiledTemplate::compile_rotated(
        &template,
        CompileConfig {
            max_levels: 4,
            coarse_step_deg: 30.0,
            min_step_deg: 7.5,
            fill_value: 0,
            precompute_coarsest: true,
        },
    )
    .unwrap();
    let matcher_rot = Matcher::new(compiled_rot).with_config(MatchConfig {
        metric: Metric::Zncc,
        rotation: RotationMode::Enabled,
        parallel: false,
        max_image_levels: 4,
        beam_width: 6,
        per_angle_topk: 3,
        roi_radius: 6,
        nms_radius: 4,
        angle_half_range_steps: 1,
        ..MatchConfig::default()
    });

    c.bench_function("zncc_masked_rotation_on", |b| {
        b.iter(|| black_box(matcher_rot.match_image(image_rot_view).unwrap()));
    });

    let matcher_ssd_rot = Matcher::new(
        CompiledTemplate::compile_rotated(
            &template,
            CompileConfig {
                max_levels: 4,
                coarse_step_deg: 30.0,
                min_step_deg: 7.5,
                fill_value: 0,
                precompute_coarsest: true,
            },
        )
        .unwrap(),
    )
    .with_config(MatchConfig {
        metric: Metric::Ssd,
        rotation: RotationMode::Enabled,
        parallel: false,
        max_image_levels: 4,
        beam_width: 6,
        per_angle_topk: 3,
        roi_radius: 6,
        nms_radius: 4,
        angle_half_range_steps: 1,
        ..MatchConfig::default()
    });

    c.bench_function("ssd_masked_rotation_on", |b| {
        b.iter(|| black_box(matcher_ssd_rot.match_image(image_rot_view).unwrap()));
    });

    if cfg!(feature = "rayon") {
        let matcher_rot_par = Matcher::new(
            CompiledTemplate::compile_rotated(
                &template,
                CompileConfig {
                    max_levels: 4,
                    coarse_step_deg: 30.0,
                    min_step_deg: 7.5,
                    fill_value: 0,
                    precompute_coarsest: true,
                },
            )
            .unwrap(),
        )
        .with_config(MatchConfig {
            metric: Metric::Zncc,
            rotation: RotationMode::Enabled,
            parallel: true,
            max_image_levels: 4,
            beam_width: 6,
            per_angle_topk: 3,
            roi_radius: 6,
            nms_radius: 4,
            angle_half_range_steps: 1,
            ..MatchConfig::default()
        });

        c.bench_function("zncc_masked_rotation_on_parallel", |b| {
            b.iter(|| black_box(matcher_rot_par.match_image(image_rot_view).unwrap()));
        });
    }
}

criterion_group!(benches, bench_matcher);
criterion_main!(benches);
