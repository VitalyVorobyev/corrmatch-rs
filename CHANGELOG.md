# CHANGELOG

## Unreleased
- Improve docs and examples.
- Continue performance hardening of rotation-enabled path.

## 0.1.0 - 2026-01-31

### Added
- Core matcher API: `Template`, `CompiledTemplate`, `Matcher`, `MatchConfig`.
- Metrics: ZNCC and SSD (SSD scores reported as negative SSE; higher is better).
- Coarse-to-fine pyramid search with optional rotation search on a discrete angle grid.
- Candidate pruning: per-angle Top‑K + 2D NMS + deterministic ordering.
- Refinement across pyramid levels plus final quadratic fits (subpixel/subangle).
- Deterministic parallel execution (feature `rayon`, opt-in via `MatchConfig.parallel`).
- Optional SIMD kernels for unmasked translation-only path (feature `simd`).
- Optional image I/O helpers (feature `image-io`).
- JSON-driven CLI (`corrmatch-cli`) with tracing support.
- Python bindings via PyO3 + maturin (`corrmatch-py`) with numpy-first API.
- Synthetic validation suite (Rust + Python).
- Criterion benchmark suite (`benches/corrmatch.rs`).

### Fixed
- Rotation-enabled matching correctness for large templates by removing `u16` index truncation in
  masked template plans.
