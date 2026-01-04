use corrmatch::lowlevel::{rotate_u8_bilinear, AngleGrid};
use corrmatch::{CompileConfig, CompiledTemplate, ImageView, Template};

#[test]
fn angle_grid_full_range_and_nearest_index() {
    let grid = AngleGrid::full(90.0).unwrap();
    assert_eq!(grid.len(), 4);

    let angles: Vec<f32> = grid.iter().collect();
    let expected = [-180.0f32, -90.0f32, 0.0f32, 90.0f32];
    for (angle, expected) in angles.iter().zip(expected.iter()) {
        assert!((angle - expected).abs() < 1e-6);
    }

    assert_eq!(grid.nearest_index(179.0), 0);
    assert_eq!(grid.nearest_index(-91.0), 1);

    let within = grid.indices_within(0.0, 95.0);
    assert_eq!(within, vec![1, 2, 3]);
}

#[test]
fn rotate_u8_bilinear_identity_and_180() {
    let width = 4;
    let height = 3;
    let data: Vec<u8> = (0u8..(width * height) as u8).collect();
    let view = ImageView::from_slice(&data, width, height).unwrap();

    let rotated = rotate_u8_bilinear(view, 0.0, 0);
    assert_eq!(rotated.data(), data.as_slice());

    let rotated_180 = rotate_u8_bilinear(view, 180.0, 0);
    let mut expected = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let src_x = width - 1 - x;
            let src_y = height - 1 - y;
            expected[y * width + x] = data[src_y * width + src_x];
        }
    }
    assert_eq!(rotated_180.data(), expected.as_slice());

    // We avoid 90-degree tests because bilinear sampling and center
    // conventions make exact integer equality ambiguous.
}

#[test]
fn rotate_constant_image_with_matching_fill() {
    let width = 5;
    let height = 4;
    let data = vec![7u8; width * height];
    let view = ImageView::from_slice(&data, width, height).unwrap();

    let rotated = rotate_u8_bilinear(view, 33.0, 7);
    assert!(rotated.data().iter().all(|&v| v == 7));
}

#[test]
fn compiled_template_caches_rotations() {
    let width = 8;
    let height = 6;
    let data: Vec<u8> = (0u8..(width * height) as u8).collect();
    let template = Template::new(data.clone(), width, height).unwrap();

    let cfg = CompileConfig {
        max_levels: 2,
        coarse_step_deg: 90.0,
        min_step_deg: 90.0,
        fill_value: 0,
        precompute_coarsest: true,
    };
    let compiled = CompiledTemplate::compile_rotated(&template, cfg).unwrap();

    let grid = compiled.angle_grid(0).unwrap();
    let idx = grid.nearest_index(0.0);
    let first = compiled.rotated_zncc_plan(0, idx).unwrap();
    let second = compiled.rotated_zncc_plan(0, idx).unwrap();

    assert_eq!(first.width(), width);
    assert_eq!(first.height(), height);
    assert!(std::ptr::eq(first, second));
    let expected_sum_w = (width - 1) as f32 * (height - 1) as f32;
    assert!((first.sum_w() - expected_sum_w).abs() < 1e-6);

    let grid1 = compiled.angle_grid(1).unwrap();
    let idx1 = grid1.nearest_index(45.0);
    let plan1 = compiled.rotated_zncc_plan(1, idx1).unwrap();
    assert_eq!(plan1.width(), width / 2);
    assert_eq!(plan1.height(), height / 2);
}
