use corrmatch::lowlevel::TemplatePlan;
use corrmatch::{
    CompileConfig, CompileConfigNoRot, CompiledTemplate, CorrMatchError, ImagePyramid, ImageView,
    Template,
};

#[test]
fn image_view_rejects_invalid_dimensions() {
    let data = [0u8; 4];

    let err = ImageView::from_slice(&data, 0, 1).err().unwrap();
    assert_eq!(
        err,
        CorrMatchError::InvalidDimensions {
            width: 0,
            height: 1,
        }
    );

    let err = ImageView::from_slice(&data, 1, 0).err().unwrap();
    assert_eq!(
        err,
        CorrMatchError::InvalidDimensions {
            width: 1,
            height: 0,
        }
    );
}

#[test]
fn image_view_rejects_invalid_stride() {
    let data = [0u8; 8];

    let err = ImageView::new(&data, 4, 1, 3).err().unwrap();
    assert_eq!(
        err,
        CorrMatchError::InvalidStride {
            width: 4,
            stride: 3,
        }
    );
}

#[test]
fn image_view_rejects_small_buffer() {
    let data = [0u8; 3];

    let err = ImageView::new(&data, 2, 2, 2).err().unwrap();
    assert_eq!(err, CorrMatchError::BufferTooSmall { needed: 4, got: 3 });
}

#[test]
fn image_view_roi_matches_expected_values() {
    let data: Vec<u8> = (0u8..16).collect();
    let view = ImageView::from_slice(&data, 4, 4).unwrap();
    assert_eq!(view.stride(), 4);
    assert_eq!(view.as_slice(), data.as_slice());

    let roi = view.roi(1, 1, 2, 2).unwrap();
    assert_eq!(roi.width(), 2);
    assert_eq!(roi.height(), 2);
    assert_eq!(roi.stride(), 4);
    assert_eq!(roi.row(0).unwrap(), &[5u8, 6u8]);
    assert_eq!(roi.row(1).unwrap(), &[9u8, 10u8]);
    assert_eq!(roi.get(0, 0).copied(), Some(5u8));
    assert!(roi.get(2, 0).is_none());

    let err = view.roi(3, 3, 2, 2).err().unwrap();
    assert_eq!(
        err,
        CorrMatchError::RoiOutOfBounds {
            x: 3,
            y: 3,
            width: 2,
            height: 2,
            img_width: 4,
            img_height: 4,
        }
    );
}

#[test]
fn image_pyramid_downsamples_by_two() {
    let data: Vec<u8> = (0u8..16).collect();
    let view = ImageView::from_slice(&data, 4, 4).unwrap();

    let pyramid = ImagePyramid::build_u8(view, 10).unwrap();
    assert_eq!(pyramid.levels().len(), 3);

    let level1 = pyramid.level(1).unwrap();
    assert_eq!(level1.width(), 2);
    assert_eq!(level1.height(), 2);
    assert_eq!(level1.row(0).unwrap(), &[3u8, 5u8]);
    assert_eq!(level1.row(1).unwrap(), &[11u8, 13u8]);

    let level2 = pyramid.level(2).unwrap();
    assert_eq!(level2.width(), 1);
    assert_eq!(level2.height(), 1);
}

#[test]
fn template_plan_matches_known_stats() {
    let tpl = Template::new(vec![0u8, 1, 2, 3], 2, 2).unwrap();
    let plan = TemplatePlan::from_view(tpl.view()).unwrap();

    let expected_mean = 1.5f32;
    let expected_inv_std = 1.0f32 / 1.25f32.sqrt();
    assert_eq!(plan.width(), 2);
    assert_eq!(plan.height(), 2);
    assert!((plan.mean() - expected_mean).abs() < 1e-6);
    assert!((plan.inv_std() - expected_inv_std).abs() < 1e-6);

    let expected_zero_mean = [-1.5f32, -0.5f32, 0.5f32, 1.5f32];
    for (value, expected) in plan.zero_mean().iter().zip(expected_zero_mean.iter()) {
        assert!((value - expected).abs() < 1e-6);
    }
}

#[test]
fn template_plan_rejects_degenerate_templates() {
    let tpl = Template::new(vec![5u8; 4], 2, 2).unwrap();
    let err = TemplatePlan::from_view(tpl.view()).err().unwrap();
    assert_eq!(
        err,
        CorrMatchError::DegenerateTemplate {
            reason: "zero variance",
        }
    );
}

#[test]
fn compiled_template_trims_degenerate_coarsest_levels() {
    let tpl_width = 4;
    let tpl_height = 4;
    let tpl_data: Vec<u8> = (0u8..16).collect();
    let template = Template::new(tpl_data, tpl_width, tpl_height).unwrap();

    let compiled =
        CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot { max_levels: 10 })
            .unwrap();

    assert_eq!(compiled.num_levels(), 2);
    let (w, h) = compiled.level_size(compiled.num_levels() - 1).unwrap();
    assert_eq!((w, h), (2, 2));
}

#[test]
fn compiled_rotated_skips_too_small_levels() {
    let tpl_width = 4;
    let tpl_height = 4;
    let tpl_data: Vec<u8> = (0u8..16).collect();
    let template = Template::new(tpl_data, tpl_width, tpl_height).unwrap();

    let compiled = CompiledTemplate::compile_rotated(
        &template,
        CompileConfig {
            max_levels: 10,
            precompute_coarsest: false,
            ..CompileConfig::default()
        },
    )
    .unwrap();

    assert!(compiled.num_levels() >= 1);
    let (w, h) = compiled.level_size(compiled.num_levels() - 1).unwrap();
    assert!(w >= 3 && h >= 3);
}
