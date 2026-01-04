# corrmatch

CorrMatch is a CPU-first template matching crate focused on ZNCC scoring across
image pyramids with hierarchical rotation search. The goal is deterministic,
reproducible matching with a minimal dependency footprint.

## Feature flags
- `rayon`: parallel search execution (off by default).
- `simd`: SIMD-accelerated kernels via `wide` (off by default).
- `image-io`: enable `image` crate helpers for examples (off by default).

## Current building blocks
```rust
use corrmatch::{CorrMatchResult, ImagePyramid, ImageView, Template, TemplatePlan};

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
Core data types are implemented; matching algorithms are not implemented yet.
