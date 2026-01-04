# corrmatch

CorrMatch is a CPU-first template matching crate focused on ZNCC scoring across
image pyramids with hierarchical rotation search. The goal is deterministic,
reproducible matching with a minimal dependency footprint.

## Feature flags
- `rayon`: parallel search execution (off by default).
- `simd`: SIMD-accelerated kernels via `wide` (off by default).
- `image-io`: enable `image` crate helpers for examples (off by default).

## Search options
- `Metric::Zncc` is supported today; `Metric::Ssd` is scaffolded but not implemented.
- `RotationMode::Disabled` (default) uses the unmasked fast path; enable rotation
  when you need angle search.
- `MatchConfig.parallel` enables rayon-backed parallel search when the feature
  is enabled; otherwise it falls back to sequential execution.

## Current building blocks
```rust
use corrmatch::bank::{CompileConfig, CompiledTemplate};
use corrmatch::search::{MatchConfig, Matcher, RotationMode};
use corrmatch::template::rotate::rotate_u8_bilinear_masked;
use corrmatch::{
    scan_masked_zncc_scalar, CorrMatchResult, ImagePyramid, ImageView,
    MaskedTemplatePlan, Peak, Template, TemplatePlan,
};

fn prepare_plan(
    image: &[u8],
    width: usize,
    height: usize,
    tpl: Vec<u8>,
    tpl_width: usize,
    tpl_height: usize,
) -> CorrMatchResult<TemplatePlan> {
    let view = ImageView::from_slice(image, width, height)?;
    let _pyramid = ImagePyramid::build_u8(view, 4)?;
    let template = Template::new(tpl, tpl_width, tpl_height)?;
    TemplatePlan::from_view(template.view())
}

fn compile_template(
    tpl: Vec<u8>,
    tpl_width: usize,
    tpl_height: usize,
) -> CorrMatchResult<CompiledTemplate> {
    let template = Template::new(tpl, tpl_width, tpl_height)?;
    CompiledTemplate::compile(&template, CompileConfig::default())
}

fn match_template(
    image: &[u8],
    width: usize,
    height: usize,
    tpl: Vec<u8>,
    tpl_width: usize,
    tpl_height: usize,
) -> CorrMatchResult<corrmatch::Match> {
    let template = Template::new(tpl, tpl_width, tpl_height)?;
    let compiled = CompiledTemplate::compile(&template, CompileConfig::default())?;
    let matcher = Matcher::new(compiled).with_config(MatchConfig {
        rotation: RotationMode::Enabled,
        ..MatchConfig::default()
    });
    let image_view = ImageView::from_slice(image, width, height)?;
    matcher.match_image(image_view)
}

fn scan_one_angle(
    image: &[u8],
    width: usize,
    height: usize,
    tpl: Vec<u8>,
    tpl_width: usize,
    tpl_height: usize,
    angle_deg: f32,
) -> CorrMatchResult<Vec<Peak>> {
    let image_view = ImageView::from_slice(image, width, height)?;
    let tpl_view = ImageView::from_slice(&tpl, tpl_width, tpl_height)?;
    let (rotated, mask) = rotate_u8_bilinear_masked(tpl_view, angle_deg, 0);
    let plan = MaskedTemplatePlan::from_rotated_u8(rotated.view(), mask, angle_deg)?;
    scan_masked_zncc_scalar(image_view, &plan, 0, 5)
}
```

## Planned API sketch
```rust
use corrmatch::CorrMatchResult;

// Pseudo-code for the planned API shape.
// let plan = corrmatch::TemplatePlan::new(template, config);
// let matcher = corrmatch::Matcher::new(plan);
// let matches = matcher.search(&image)?;
```

## Status
Core data types, compiled template assets, a coarse-to-fine matcher, and
subpixel/subangle refinement are implemented; higher-level APIs and
SIMD/parallel acceleration are pending.

## Benchmarks
Run the benchmark suite with:
```
cargo bench
```
To enable the parallel path:
```
cargo bench --features rayon
```
You can also run the test suite with rayon enabled:
```
cargo test --features rayon
```
