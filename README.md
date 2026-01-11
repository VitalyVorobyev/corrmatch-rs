# corrmatch
[![CI](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/ci.yml)
[![Python Tests](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/test_py.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/test_py.yml)
[![Security Audit](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/audit.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/corrmatch-rs/actions/workflows/audit.yml)
[![crates.io](https://img.shields.io/crates/v/corrmatch.svg)](https://crates.io/crates/corrmatch)
[![docs.rs](https://img.shields.io/docsrs/corrmatch)](https://docs.rs/corrmatch)
[![license](https://img.shields.io/crates/l/corrmatch.svg)](LICENSE)

CorrMatch is a CPU-first template matching library for grayscale images. It
implements a coarse-to-fine pyramid search with optional rotation and two
metrics: ZNCC and SSD. The focus is deterministic, reproducible matching with
minimal dependencies.

## Install
```toml
[dependencies]
corrmatch = "0.1"
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
matcher.match_image_topk(image_view, 5)
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

The schema lives at `corrmatch-cli/config.schema.json`, and an example config is
at `corrmatch-cli/config.example.json`.

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
- `simd`: SIMD-accelerated kernels (planned).
- `image-io`: file I/O helpers via the `image` crate.

## Python bindings (corrmatch-py)
The workspace includes PyO3 bindings in `corrmatch-py`.

- Build locally: `cd corrmatch-py && maturin develop --release`
- Run tests: `python -m pytest python/tests`

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

## Status
Core matcher types, the JSON-driven CLI, and Python bindings are implemented.
See `ROADMAP.md` for upcoming milestones.
