# corrmatch

CorrMatch is a CPU-first template matching crate focused on ZNCC scoring across
image pyramids with hierarchical rotation search. The goal is deterministic,
reproducible matching with a minimal dependency footprint.

## Feature flags
- `rayon`: parallel search execution (off by default).
- `simd`: SIMD-accelerated kernels via `wide` (off by default).
- `image-io`: enable `image` crate helpers for examples (off by default).

## Planned API sketch
```rust
use corrmatch::{CorrMatchError, Result};

// Pseudo-code for the planned API shape.
// let plan = corrmatch::TemplatePlan::new(template, config);
// let matcher = corrmatch::Matcher::new(plan);
// let matches = matcher.search(&image)?;
```

## Status
Scaffolding; algorithms not implemented yet.
