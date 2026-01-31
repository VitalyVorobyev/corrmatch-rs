use corrmatch::lowlevel::score_masked_zncc_at;
use corrmatch::{CompileConfig, CompiledTemplate, ImageView, Template};

fn make_template(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v = (x as u32)
                .wrapping_mul(13)
                .wrapping_add((y as u32).wrapping_mul(7))
                .wrapping_add((x as u32).wrapping_mul(y as u32));
            data.push((v & 0xFF) as u8);
        }
    }
    data
}

#[test]
fn masked_zncc_score_is_correct_for_large_template() {
    let tpl_width = 512;
    let tpl_height = 512;
    let tpl_data = make_template(tpl_width, tpl_height);
    let template = Template::new(tpl_data.clone(), tpl_width, tpl_height).unwrap();

    let img_width = 520;
    let img_height = 520;
    let x0 = 3;
    let y0 = 5;
    let mut image = vec![0u8; img_width * img_height];
    for y in 0..tpl_height {
        let src = &tpl_data[y * tpl_width..(y + 1) * tpl_width];
        let dst = &mut image[(y0 + y) * img_width + x0..(y0 + y) * img_width + x0 + tpl_width];
        dst.copy_from_slice(src);
    }
    let image_view = ImageView::from_slice(&image, img_width, img_height).unwrap();

    let compiled = CompiledTemplate::compile_rotated(
        &template,
        CompileConfig {
            max_levels: 1,
            coarse_step_deg: 180.0,
            min_step_deg: 180.0,
            fill_value: 0,
            precompute_coarsest: true,
        },
    )
    .unwrap();

    let grid = compiled.angle_grid(0).unwrap();
    let angle_idx = grid.nearest_index(0.0);
    let plan = compiled.rotated_zncc_plan(0, angle_idx).unwrap();

    let score = score_masked_zncc_at(image_view, plan, x0, y0, 1e-8);
    assert!(score.is_finite());
    assert!(score > 0.999, "expected score near 1, got {score}");
}
