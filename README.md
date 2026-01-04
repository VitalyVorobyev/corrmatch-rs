# corrmatch

CorrMatch is a CPU-first template matching crate focused on ZNCC/SSD scoring
across image pyramids with hierarchical rotation search. The goal is
deterministic, reproducible matching with a minimal dependency footprint.

## Feature flags
- `rayon`: parallel search execution (off by default).
- `simd`: SIMD-accelerated kernels via `wide` (off by default).
- `image-io`: enable `image` crate helpers and file I/O (off by default).

## Search options
- `Metric::Zncc` and `Metric::Ssd` are supported.
- `RotationMode::Disabled` (default) uses the unmasked fast path; enable rotation
  when you need angle search.
- `MatchConfig.parallel` enables rayon-backed parallel search when the feature
  is enabled; otherwise it falls back to sequential execution.

## Quickstart
```rust
use corrmatch::bank::{CompileConfig, CompileConfigNoRot, CompiledTemplate};
use corrmatch::search::{MatchConfig, Matcher, RotationMode};
use corrmatch::{CorrMatchResult, ImageView, Template};

fn match_template(
    image: &[u8],
    width: usize,
    height: usize,
    tpl: Vec<u8>,
    tpl_width: usize,
    tpl_height: usize,
) -> CorrMatchResult<corrmatch::Match> {
    let template = Template::new(tpl, tpl_width, tpl_height)?;
    let compiled = template.compile(CompileConfig::default())?;
    let matcher = Matcher::new(compiled).with_config(MatchConfig {
        rotation: RotationMode::Enabled,
        ..MatchConfig::default()
    });
    let image_view = ImageView::from_slice(image, width, height)?;
    matcher.match_image(image_view)
}
```

To retrieve multiple results:
```rust
# use corrmatch::bank::CompileConfig;
# use corrmatch::search::{MatchConfig, Matcher, RotationMode};
# use corrmatch::{CorrMatchResult, ImageView, Template};
# fn match_topk(image: &[u8], w: usize, h: usize, tpl: Vec<u8>, tw: usize, th: usize) -> CorrMatchResult<Vec<corrmatch::Match>> {
let template = Template::new(tpl, tw, th)?;
let compiled = template.compile(CompileConfig::default())?;
let matcher = Matcher::new(compiled).with_config(MatchConfig {
    rotation: RotationMode::Enabled,
    ..MatchConfig::default()
});
let image_view = ImageView::from_slice(image, w, h)?;
matcher.match_image_topk(image_view, 5)
# }
```

If you do not need rotation support, compile a lighter template:
```rust
# use corrmatch::bank::{CompileConfigNoRot, CompiledTemplate};
# use corrmatch::{CorrMatchResult, Template};
# fn compile_no_rot(tpl: Vec<u8>, w: usize, h: usize) -> CorrMatchResult<CompiledTemplate> {
let template = Template::new(tpl, w, h)?;
CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot::default())
# }
```

## Low-level building blocks
Advanced hooks live in `corrmatch::lowlevel`:
```rust
use corrmatch::lowlevel::{scan_masked_zncc_scalar, MaskedTemplatePlan, Peak};
use corrmatch::template::rotate::rotate_u8_bilinear_masked;
use corrmatch::{CorrMatchResult, ImageView, Template};

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

## Image I/O (feature `image-io`)
```rust
# #[cfg(feature = "image-io")]
# {
use corrmatch::image::io::load_gray_image;
let template = load_gray_image("template.png")?;
# }
```

## Status
Core data types, compiled template assets, a coarse-to-fine matcher, and
subpixel/subangle refinement are implemented; SIMD acceleration and broader
API ergonomics are pending.

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
