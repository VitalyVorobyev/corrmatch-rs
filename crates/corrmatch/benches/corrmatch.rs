use corrmatch::lowlevel::rotate_u8_bilinear_masked;
use corrmatch::lowlevel::{scan_masked_zncc_scalar_full, MaskedTemplatePlan};
use corrmatch::{
    CompileConfig, CompileConfigNoRot, CompiledTemplate, ImageView, MatchConfig, Matcher, Metric,
    RotationMode, Template,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

fn norot_cfg(max_levels: usize) -> CompileConfigNoRot {
    let mut c = CompileConfigNoRot::default();
    c.max_levels = max_levels;
    c
}

fn rot_cfg(max_levels: usize, coarse_step_deg: f32, min_step_deg: f32) -> CompileConfig {
    let mut c = CompileConfig::default();
    c.max_levels = max_levels;
    c.coarse_step_deg = coarse_step_deg;
    c.min_step_deg = min_step_deg;
    c
}

fn match_cfg_unmasked(metric: Metric, parallel: bool, max_image_levels: usize) -> MatchConfig {
    let mut c = MatchConfig::default();
    c.metric = metric;
    c.rotation = RotationMode::Disabled;
    c.parallel = parallel;
    c.max_image_levels = max_image_levels;
    c.beam_width = 6;
    c.per_angle_topk = 3;
    c.roi_radius = 6;
    c.nms_radius = 4;
    c.angle_half_range_steps = 1;
    c
}

fn match_cfg_masked(metric: Metric, parallel: bool, max_image_levels: usize) -> MatchConfig {
    let mut c = MatchConfig::default();
    c.metric = metric;
    c.rotation = RotationMode::Enabled;
    c.parallel = parallel;
    c.max_image_levels = max_image_levels;
    c.beam_width = 6;
    c.per_angle_topk = 3;
    c.roi_radius = 6;
    c.nms_radius = 4;
    c.angle_half_range_steps = 1;
    c
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

    let compiled_unmasked = CompiledTemplate::compile_unrotated(&template, norot_cfg(4)).unwrap();
    let matcher_unmasked =
        Matcher::new(compiled_unmasked).with_config(match_cfg_unmasked(Metric::Zncc, false, 4));

    c.bench_function("zncc_unmasked_rotation_off", |b| {
        b.iter(|| black_box(matcher_unmasked.match_image(image_view).unwrap()));
    });

    let matcher_ssd_unmasked =
        Matcher::new(CompiledTemplate::compile_unrotated(&template, norot_cfg(4)).unwrap())
            .with_config(match_cfg_unmasked(Metric::Ssd, false, 4));

    c.bench_function("ssd_unmasked_rotation_off", |b| {
        b.iter(|| black_box(matcher_ssd_unmasked.match_image(image_view).unwrap()));
    });

    if cfg!(feature = "rayon") {
        let matcher_unmasked_par =
            Matcher::new(CompiledTemplate::compile_unrotated(&template, norot_cfg(4)).unwrap())
                .with_config(match_cfg_unmasked(Metric::Zncc, true, 4));

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

    let compiled_rot = CompiledTemplate::compile_rotated(&template, rot_cfg(4, 30.0, 7.5)).unwrap();
    let matcher_rot =
        Matcher::new(compiled_rot).with_config(match_cfg_masked(Metric::Zncc, false, 4));

    c.bench_function("zncc_masked_rotation_on", |b| {
        b.iter(|| black_box(matcher_rot.match_image(image_rot_view).unwrap()));
    });

    let matcher_ssd_rot =
        Matcher::new(CompiledTemplate::compile_rotated(&template, rot_cfg(4, 30.0, 7.5)).unwrap())
            .with_config(match_cfg_masked(Metric::Ssd, false, 4));

    c.bench_function("ssd_masked_rotation_on", |b| {
        b.iter(|| black_box(matcher_ssd_rot.match_image(image_rot_view).unwrap()));
    });

    if cfg!(feature = "rayon") {
        let matcher_rot_par = Matcher::new(
            CompiledTemplate::compile_rotated(&template, rot_cfg(4, 30.0, 7.5)).unwrap(),
        )
        .with_config(match_cfg_masked(Metric::Zncc, true, 4));

        c.bench_function("zncc_masked_rotation_on_parallel", |b| {
            b.iter(|| black_box(matcher_rot_par.match_image(image_rot_view).unwrap()));
        });
    }
}

/// Benchmarks for kernel-level dispatch: masked vs unmasked ZNCC, scalar vs integral path.
///
/// The ZNCC integral path is exercised when rotation is disabled, the metric is ZNCC,
/// and `parallel` is false. Enabling `parallel` forces the plain scalar path even
/// without the rayon feature (falls back to scalar), which allows a sequential
/// apples-to-apples comparison.
fn bench_kernel_dispatch(c: &mut Criterion) {
    let img_width = 512;
    let img_height = 512;
    let image = make_image(img_width, img_height);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();

    let tpl_width = 48;
    let tpl_height = 48;
    let tpl_x0 = 100;
    let tpl_y0 = 80;
    let tpl_data = extract_patch(&image, img_width, tpl_x0, tpl_y0, tpl_width, tpl_height);
    let tpl_view = ImageView::from_slice(&tpl_data, tpl_width, tpl_height).unwrap();

    // Masked ZNCC plan (all pixels valid = unmasked equivalent).
    let full_mask: Vec<u8> = vec![1u8; tpl_width * tpl_height];
    let masked_plan = MaskedTemplatePlan::from_rotated_u8(tpl_view, full_mask, 0.0)
        .expect("valid masked template plan");

    let mut group = c.benchmark_group("kernel_dispatch");

    // Masked ZNCC scalar (via lowlevel scan function).
    group.bench_function("masked_zncc_scalar", |b| {
        b.iter(|| {
            black_box(
                scan_masked_zncc_scalar_full(
                    black_box(image_view),
                    black_box(&masked_plan),
                    0,
                    5,
                    1e-8,
                    f32::NEG_INFINITY,
                )
                .unwrap(),
            )
        });
    });

    // Unmasked ZNCC via Matcher — sequential, rotation disabled.
    // The MatchConfig selects the ZNCC-integral path internally.
    let compiled_norot = CompiledTemplate::compile_unrotated(
        &Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap(),
        norot_cfg(1),
    )
    .unwrap();
    let matcher_zncc_seq = Matcher::new(compiled_norot).with_config({
        let mut cfg = MatchConfig::default();
        cfg.metric = Metric::Zncc;
        cfg.rotation = RotationMode::Disabled;
        cfg.parallel = false;
        cfg.max_image_levels = 1;
        cfg.beam_width = 5;
        cfg.per_angle_topk = 5;
        cfg
    });
    group.bench_function("unmasked_zncc_integral_seq", |b| {
        b.iter(|| black_box(matcher_zncc_seq.match_image(black_box(image_view)).unwrap()));
    });

    // Unmasked SSD via Matcher — sequential.
    let compiled_ssd = CompiledTemplate::compile_unrotated(
        &Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap(),
        norot_cfg(1),
    )
    .unwrap();
    let matcher_ssd_seq = Matcher::new(compiled_ssd).with_config({
        let mut cfg = MatchConfig::default();
        cfg.metric = Metric::Ssd;
        cfg.rotation = RotationMode::Disabled;
        cfg.parallel = false;
        cfg.max_image_levels = 1;
        cfg.beam_width = 5;
        cfg.per_angle_topk = 5;
        cfg
    });
    group.bench_function("unmasked_ssd_scalar_seq", |b| {
        b.iter(|| black_box(matcher_ssd_seq.match_image(black_box(image_view)).unwrap()));
    });

    group.finish();
}

/// Benchmarks sequential vs parallel execution on a 1024×1024 image.
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let img_width = 1024;
    let img_height = 1024;
    let image = make_image(img_width, img_height);
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();

    let tpl_width = 64;
    let tpl_height = 64;
    let tpl_x0 = 200;
    let tpl_y0 = 180;
    let tpl_data = extract_patch(&image, img_width, tpl_x0, tpl_y0, tpl_width, tpl_height);
    let template = Template::new(tpl_data, tpl_width, tpl_height).unwrap();

    let compiled = CompiledTemplate::compile_unrotated(&template, norot_cfg(4)).unwrap();
    let mut group = c.benchmark_group("parallel_vs_sequential");

    let cfg_seq = match_cfg_unmasked(Metric::Zncc, false, 4);
    let matcher_seq =
        Matcher::new(CompiledTemplate::compile_unrotated(&template, norot_cfg(4)).unwrap())
            .with_config(cfg_seq);
    group.bench_with_input(
        BenchmarkId::new("zncc_unmasked", "sequential"),
        &(),
        |b, _| {
            b.iter(|| black_box(matcher_seq.match_image(black_box(image_view)).unwrap()));
        },
    );

    if cfg!(feature = "rayon") {
        let cfg_par = match_cfg_unmasked(Metric::Zncc, true, 4);
        let matcher_par =
            Matcher::new(CompiledTemplate::compile_unrotated(&template, norot_cfg(4)).unwrap())
                .with_config(cfg_par);
        group.bench_with_input(
            BenchmarkId::new("zncc_unmasked", "parallel"),
            &(),
            |b, _| {
                b.iter(|| black_box(matcher_par.match_image(black_box(image_view)).unwrap()));
            },
        );
    }

    group.finish();

    // Suppress unused-variable warning on `compiled` when rayon is disabled.
    let _ = compiled;
}

criterion_group!(
    benches,
    bench_matcher,
    bench_kernel_dispatch,
    bench_parallel_vs_sequential
);
criterion_main!(benches);
