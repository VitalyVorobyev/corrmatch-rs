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
- CLI tool (corrmatch-cli)
- Synthetic case generator (tools/synth_cases)
- Code review and bug fixes (rotation bounds, config validation)

**In Progress:**
- Milestone 7 (docs/examples/benches) - expand examples and document benchmark usage
- Milestone 6 SIMD kernels (partial) - validate parity and tolerances

**Next Steps:**
1. Finish SIMD kernels and document floating tolerances (Milestone 6)
2. Publish runnable examples and benchmark docs (Milestone 7)
3. Prepare Python packaging/publishing workflow
