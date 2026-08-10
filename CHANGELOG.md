# CHANGELOG

## Unreleased

No entries yet. New changes are recorded here until the next version tag is cut.

---

## 0.2.1 - 2026-08-10

Chore release: update dependencies

## 0.2.0 - 2026-04-15

### Breaking Changes

- **Workspace restructure** (F21): library, CLI, and Python bindings moved into
  `crates/corrmatch`, `crates/corrmatch-cli`, and `crates/corrmatch-py` respectively.
  The root `Cargo.toml` is now workspace-only.

- **`#[non_exhaustive]`** (F1): `CorrMatchError`, `Metric`, `RotationMode`,
  `MatchConfig`, `CompileConfig`, and `CompileConfigNoRot` are now marked
  `#[non_exhaustive]`. External callers must use `Default::default()` and
  field mutation (or provided constructors) instead of struct literal syntax.
  Matching on `Metric` and `RotationMode` requires a `_` wildcard arm.

- **Opaque `CompiledTemplate`** (F2): `CompiledTemplate` is now an opaque
  `pub struct` (previously a `pub enum`). The inner types `CompiledTemplateRot`
  and `CompiledTemplateNoRot` are `pub(crate)`. Use `is_rotated()` instead of
  pattern matching on the variants. `CompiledTemplate::compile` (the alias for
  `compile_rotated`) has been removed; use `compile_rotated` directly.

- **`ValidCoord` fields promoted to `u32`** (F13): `ValidCoord { x: u32, y: u32 }`
  (previously `u16`). Code in `lowlevel` that reads `.x` or `.y` from a
  `ValidCoord` value may need type-cast updates.

- **Version bumped to `0.2.0`** across all three crates and `pyproject.toml`.

- **MSRV bumped from 1.70 to 1.88.** The Cargo.lock file is now written in
  v4 format which older toolchains cannot parse, and transitive
  dependencies (notably `moxcms`, pulled in via `image`) require a recent
  cargo/rustc to parse their manifests. 1.88 (June 2025) covers both
  constraints without forcing downstream users onto a bleeding-edge
  toolchain.

### Added

- `CompileConfigNoRot::validate()` method (F7).
- `CompiledTemplate::is_rotated()` public method.
- `Matcher::from_arc(Arc<CompiledTemplate>)` constructor enabling reusable
  matchers from a single compiled template (F11). The Python
  `CompiledTemplate.matcher(cfg)` is now idempotent and can be called
  repeatedly with different configs without recompiling the template.
- MSRV CI job in `.github/workflows/ci.yml` (F4).
- Python-native API layer (F3): `corrmatch.Match`, `corrmatch.MatchConfig`,
  and `corrmatch.CompileConfig` are now `@dataclass(frozen=True, kw_only=True)`
  wrappers with `Literal["zncc", "ssd"]` typing, full Google-style
  docstrings, construction-time validation, and PEP 561 stubs
  (`__init__.pyi`, `_corrmatch.pyi`). Raw PyO3 types remain available under
  `corrmatch._corrmatch`. Type-checked with `mypy --strict` in CI.
- Python test matrix expanded to include 3.10 and 3.12 (F14).
- Targeted Criterion benchmark groups for kernel dispatch and
  parallel-vs-sequential (F19).

### Fixed

- `release-pypi.yml` previously referenced a non-existent
  `crates/corrmatch-py/Cargo.toml`; the workspace restructure (F21)
  unbreaks the workflow.
- Removed the stale `RUSTSEC-2024-0436` ignore in `audit.yml` — the
  affected crate (`paste`) is no longer in the dependency tree (F6).
- Asymmetric rotation boundary handling between `rotate_u8_bilinear` and
  `rotate_u8_bilinear_masked` is now factored into a shared helper and
  documented (F12).

### Removed

- `TemplatePlan::t_prime()` (alias for `zero_mean()`) (F8).
- `CompiledTemplate::compile` (alias for `compile_rotated`) (F10).
- Dead `refine_to_finer_level` function in `search/refine.rs` (F9).

---

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
