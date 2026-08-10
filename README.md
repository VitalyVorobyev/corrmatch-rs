<h1>
  <a href="https://vitavision.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="book/images/vv-favicon-dark.svg">
      <img src="book/images/vv-favicon-dark.svg" alt="vitavision.dev" height="48" align="left">
    </picture>
  </a>
  &nbsp;corrmatch
</h1>

[![CI](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/ci.yml)
[![Python Tests](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/test_py.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/test_py.yml)
[![Security Audit](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/audit.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/audit.yml)
[![crates.io](https://img.shields.io/crates/v/corrmatch.svg)](https://crates.io/crates/corrmatch)
[![docs.rs](https://img.shields.io/docsrs/corrmatch)](https://docs.rs/corrmatch)
[![license](https://img.shields.io/crates/l/corrmatch.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.91-blue.svg)](https://blog.rust-lang.org/2025/10/30/Rust-1.91.0/)

CorrMatch is a CPU-first template matching library for grayscale images. It
implements a coarse-to-fine pyramid search with optional rotation and two
metrics: ZNCC and SSD. The focus is deterministic, reproducible matching with
minimal dependencies.

## Install
```toml
[dependencies]
corrmatch = "0.2"
```

## Quickstart (library)
```rust
use corrmatch::{
    CompileConfig, MatchConfig, Matcher, RotationMode, Template, ImageView,
};

# fn run(image: &[u8], width: usize, height: usize, tpl: Vec<u8>, tw: usize, th: usize)
#     -> corrmatch::CorrMatchResult<corrmatch::Match> {
let template = Template::new(tpl, tw, th)?;
let compiled = template.compile(CompileConfig::default())?;
let matcher = Matcher::new(compiled).with_config(MatchConfig {
    rotation: RotationMode::Enabled,
    ..MatchConfig::default()
});
let image_view = ImageView::from_slice(image, width, height)?;
matcher.match_image(image_view)
# }
```

### Match results (`match_image` vs `match_image_topk`)

- `Matcher::match_image(image)` returns the single best `Match`.
- `Matcher::match_image_topk(image, k)` returns up to `k` matches **sorted best-first** by `score`.

`Match` fields:
- `x`, `y`: top-left placement coordinates in the original image (level 0). These are `f32` because the final refinement can be subpixel.
- `angle_deg`: rotation angle in degrees (**clockwise**, with image coordinates x→ right, y↓ down). When rotation is disabled this is `0.0`.
- `score`: ZNCC is roughly `[-1, 1]` (higher is better). SSD is reported as negative SSE (higher is better).

To retrieve multiple results:
```rust
# use corrmatch::{CompileConfig, MatchConfig, Matcher, RotationMode, Template, ImageView};
# fn run(image: &[u8], width: usize, height: usize, tpl: Vec<u8>, tw: usize, th: usize)
#     -> corrmatch::CorrMatchResult<Vec<corrmatch::Match>> {
let template = Template::new(tpl, tw, th)?;
let compiled = template.compile(CompileConfig::default())?;
let matcher = Matcher::new(compiled).with_config(MatchConfig {
    rotation: RotationMode::Enabled,
    ..MatchConfig::default()
});
let image_view = ImageView::from_slice(image, width, height)?;
let matches = matcher.match_image_topk(image_view, 5)?;
for (i, m) in matches.iter().enumerate() {
    println!("{i}: x={}, y={}, angle_deg={}, score={}", m.x, m.y, m.angle_deg, m.score);
}
Ok(matches)
# }
```

If you do not need rotation support, compile a lighter template:
```rust
# use corrmatch::{CompileConfigNoRot, CompiledTemplate, CorrMatchResult, Template};
# fn compile_no_rot(tpl: Vec<u8>, w: usize, h: usize) -> CorrMatchResult<CompiledTemplate> {
let template = Template::new(tpl, w, h)?;
CompiledTemplate::compile_unrotated(&template, CompileConfigNoRot::default())
# }
```

## CLI (corrmatch-cli)
The workspace includes a JSON-driven CLI.

- Build: `cargo build -p corrmatch-cli`
- Run: `cargo run -p corrmatch-cli -- --config config.json`
- Print schema: `cargo run -p corrmatch-cli -- --print-schema`
- Print example: `cargo run -p corrmatch-cli -- --print-example`

The schema lives at `crates/corrmatch-cli/config.schema.json`, and an example
config is at `crates/corrmatch-cli/config.example.json`.

## Concepts
- `Template`: owned template pixels (contiguous grayscale).
- `CompiledTemplate`: precomputed template pyramid plus optional angle banks.
- `Matcher`: runs coarse-to-fine search using `MatchConfig`.
- `Metric`: `Zncc` or `Ssd`.
- `RotationMode`: `Disabled` (fast path) or `Enabled` (masked rotation search).
- Coordinates: results are top-left placement coordinates at level 0.

## Configuration
- `CompileConfig` controls template pyramid depth and rotation grid. When
  rotation is disabled, only `max_levels` is used.
- `MatchConfig` controls the search strategy (beam width, ROI size, NMS radius,
  and angle neighborhood). For SSD, `min_var_i` is ignored.

## Feature flags
- `rayon`: parallel search execution.
- `simd`: SIMD-accelerated kernels (currently: unmasked translation-only path).
- `image-io`: file I/O helpers via the `image` crate.

## Python bindings (corrmatch-py)
The workspace includes PyO3 bindings in `corrmatch-py`.

- Build locally: `cd crates/corrmatch-py && maturin develop --release`
- Run tests: `python -m pytest crates/corrmatch-py/python/tests`

Python API notes:
- `Matcher.match_image(image)` returns a `corrmatch.Match` with fields `x`, `y`, `angle_deg`, `score`.
- `Matcher.match_topk(image, k)` returns a Python `list[corrmatch.Match]` sorted best-first by `score`.
- Angle convention matches Rust: positive angles are **clockwise**.

Example (match + visualize):
```python
import numpy as np
from PIL import Image

import corrmatch
import corrmatch.viz as viz

image = np.asarray(Image.open("synthetic_cases/rotation_fine_22_5deg/image.png").convert("L"), dtype=np.uint8)
template = np.asarray(Image.open("synthetic_cases/rotation_fine_22_5deg/template.png").convert("L"), dtype=np.uint8)

compile_cfg = corrmatch.CompileConfig(max_levels=4, coarse_step_deg=30.0, min_step_deg=7.5)
match_cfg = corrmatch.MatchConfig(rotation="enabled", metric="zncc")

tpl = corrmatch.Template(template)
compiled = tpl.compile(compile_cfg)
matcher = compiled.matcher(match_cfg)

best = matcher.match_image(image)
top = matcher.match_topk(image, k=3)  # sorted best-first

viz.show_matches(image, template, top, compile_cfg=compile_cfg, match_cfg=match_cfg)
print("best:", best.x, best.y, best.angle_deg, best.score)
```

## Low-level API
Advanced hooks live in `corrmatch::lowlevel`, including template plans, kernel
traits, scan helpers, and rotation utilities. These are intended for custom
pipelines and experimentation.

## Image I/O (feature `image-io`)
```rust
# #[cfg(feature = "image-io")]
# {
use corrmatch::io::load_gray_image;
let template = load_gray_image("template.png")?;
# }
```

## Benchmarks and tests
- `cargo test`
- `cargo test --features rayon`
- `cargo bench`

## Validation and performance tracking
- Synthetic validation suite (ground truth cases): `docs/VALIDATION.md`
- Criterion benchmarks and latest numbers: `performance.md`
- Release checklist (crates.io + PyPI): `docs/RELEASE_CHECKLIST.md`

## Synthetic case examples

This repo includes ground-truth synthetic cases in `synthetic_cases/`.

Note: the per-case `cli_config.json` uses relative paths like `image.png`, so run
the CLI from inside the case directory (or use the Python helper below).

Quick visual impressions (templates + saved overlays in `book/images/`):

| Case | Template | Detection overlay |
| --- | --- | --- |
| `blur_sigma_1_5` | <img src="synthetic_cases/blur_sigma_1_5/template.png" width="140" /> | <img src="book/images/blur_sigma_1_5.png" width="520" /> |
| `rotation_fine_22_5deg` | <img src="synthetic_cases/rotation_fine_22_5deg/template.png" width="140" /> | <img src="book/images/rotation_fine_22_5deg.png" width="520" /> |
| `distractors_topk` | <img src="synthetic_cases/distractors_topk/template.png" width="140" /> | <img src="book/images/distractors_topk.png" width="520" /> |

### `blur_sigma_1_5` (translation-only, blurred)
- CLI:
  - `(cd synthetic_cases/blur_sigma_1_5 && cargo run --manifest-path ../../Cargo.toml -p corrmatch-cli -- --config cli_config.json)`
- Visualize (requires `corrmatch-py` installed):
  - `./tools/viz_case.sh blur_sigma_1_5 1`

### `rotation_fine_22_5deg` (rotation enabled)
- CLI:
  - `(cd synthetic_cases/rotation_fine_22_5deg && cargo run --manifest-path ../../Cargo.toml -p corrmatch-cli -- --config cli_config.json)`
- Visualize:
  - `./tools/viz_case.sh rotation_fine_22_5deg 1`

### `distractors_topk` (multiple candidates)
- CLI:
  - `(cd synthetic_cases/distractors_topk && cargo run --manifest-path ../../Cargo.toml -p corrmatch-cli -- --config cli_config.json)`
- Visualize top-4 overlays:
  - `./tools/viz_case.sh distractors_topk 4`

## Status
Core matcher types, the JSON-driven CLI, and Python bindings are implemented.
See `ROADMAP.md` for upcoming milestones.
