# ROADMAP

## Motivation and target use case
CorrMatch targets CPU-first template matching with ZNCC scoring and a
pyramid + hierarchical rotation search. The intended use case is fast,
deterministic matching for inspection, alignment, and registration
workloads where GPU dependencies are undesirable.

## Defaults and assumptions
- Default rotation range is full [-180, 180) degrees and must be configurable.
- Typical template sizes are 64..512 pixels on a side; common size ~256.

## Milestones

### 0. Scaffolding and crate layout
Deliverables: crate skeleton, module layout, feature flags, docs, error types.
Acceptance: `cargo test` passes and documentation files exist.

### 1. Coarsest seeding over (x, y, theta)
Deliverables: angle bank, coarse pyramid levels, ZNCC scoring for seeds.
Acceptance: seeded candidates cover (x, y, theta) grid and rank deterministically.

### 2. Beam / Top-K candidates and NMS
Deliverables: top-K tracking, spatial + angular NMS, stable ordering.
Acceptance: candidate pool size is bounded and reproducible across runs.

### 3. Refinement up the pyramid
Deliverables: ROI-based refinement per level with local angle neighborhoods.
Acceptance: coarse seeds converge to improved matches across levels.

### 4. Final quadratic fits
Deliverables: 2D quadratic fit for (x, y) and 1D quadratic fit for theta.
Acceptance: subpixel/subangle estimates meet error targets on synthetic tests.

### 5. Performance hardening
Deliverables: profiling, hot-loop tightening, allocation audits, cache-aware data.
Acceptance: benchmarks show clear speedups without accuracy regressions.

### 6. Feature-gated rayon + simd
Deliverables: parallel search via rayon and SIMD kernels via wide.
Acceptance: results match scalar within documented floating tolerance.

### 7. Docs, examples, and benches
Deliverables: runnable examples, image-io optional path, criterion benches.
Acceptance: examples and benches are reproducible and documented.

### 8. Python bindings via PyO3
Deliverables: `corrmatch-py` crate, high-level Python API, maturin build.
Acceptance: `import corrmatch` works with numpy arrays for image/template.

### 9. Synthetic test validation
Deliverables: ground-truth synthetic cases, Rust and Python validation suites.
Acceptance: all generated cases pass within tolerance (position ±3px, angle ±2°).

### 10. SSD metric support
Deliverables: Sum of Squared Differences as alternative to ZNCC.
Acceptance: SSD produces valid matches and passes synthetic tests.

## v1 scope
- CPU-only template matching with ZNCC and SSD scoring.
- Coarse-to-fine pyramid search with hierarchical rotation refinement.
- Top-K + NMS candidate pruning and quadratic final fits.
- Optional rayon and SIMD acceleration behind feature flags.
- Python bindings for numpy-based workflows.
- Clear docs, examples, and benchmarks for typical template sizes.

## Current status (2026-01)

**Completed:**
- Milestones 0-6 (core library, pyramids, refinement, deterministic rayon)
- Milestone 8 (Python bindings via PyO3 + maturin)
- Milestone 9 (synthetic validation in Rust + Python)
- Milestone 10 (SSD metric)
- CLI tool with tracing support (`--trace` flag)
- Synthetic case generator (tools/synth_cases)
- Rayon parallel rotation precomputation
- SIMD kernels (via `wide` crate - deferred, not beneficial for current bottlenecks)
- Phase 4.2: Precomputed valid indices for masked kernels (eliminates mask branch mispredictions)
- Phase 4.3: Multi-angle batch refinement for cache locality

**In Progress:**
- Milestone 5 (Performance hardening) - profiling and validation
- Milestone 7 (docs/examples/benches) - documentation and benchmarks

**Next Steps:**
1. Profile and validate performance improvements with tracing
2. Document benchmark results and performance characteristics
3. Publish runnable examples and benchmark docs (Milestone 7)
4. Prepare Python packaging/publishing workflow

**Performance Notes:**
- SIMD implementation via `wide` crate is available but deferred. Profiling showed
  u8→f32 conversion overhead and horizontal sum inefficiency make it slower than
  scalar for the current workload. The main bottleneck is masked kernels (rotation-
  enabled path), which account for ~63% of runtime at level 0 refinement.
- Precomputed valid indices (Phase 4.2) eliminates ~30-50% branch mispredictions
  from mask checks in hot loops.
- Multi-angle batch refinement (Phase 4.3) caches image rows and reuses them across
  all angle evaluations at each position, eliminating redundant memory fetches.
