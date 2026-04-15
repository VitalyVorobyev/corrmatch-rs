# CorrMatch Code Review

**Date:** 2026-01-11
**Scope:** Full library review with focus on correctness, design, and structure
**Overall Assessment:** Production-quality with minor issues to address

---

## Status update (2026-04-15) — REVIEW.md supersedes this document

This document is retained as a historical record. **REVIEW.md** (repo root) is
the authoritative tracker for all open work. All items below have been triaged
against REVIEW.md findings. See the per-item notes under each heading.

Addressed since 2026-01-11 (original 2026-01-31 update):
- Rotation casting UB: fixed by clamping source coordinates before casting.
- Config validation: `CompileConfig::validate()` and `MatchConfig::validate()` are implemented and used.
- Parallel flag behavior: requesting parallel without `rayon` now errors.
- Masked rotation plans: fixed a correctness bug for large templates by removing `u16` index truncation.
- Benches: Criterion bench is wired via `harness = false`.

Additional items resolved in 0.2.0 (2026-04-15):
- `TemplatePlan::t_prime()` removed; `zero_mean()` is now the only accessor. → **Closed** (REVIEW.md F8).
- `CompiledTemplate::compile` alias removed; `compile_rotated`/`compile_unrotated` are canonical. → **Closed** (REVIEW.md F10).
- `ValidCoord` changed from `u16` to `u32`. → **Closed** (REVIEW.md F13).
- `CompileConfigNoRot::validate()` added. → **Closed** (REVIEW.md F7).
- Rotation boundary asymmetry documented in both `rotate_u8_bilinear` and `rotate_u8_bilinear_masked`. → **Closed** (REVIEW.md F12).
- `.expect()` invariant comments added to all kernel/pyramid/rotate sites. → **Closed** (REVIEW.md F17).

Remaining items and their disposition:
- Item 6 (hardcoded `1e-8` threshold): **WONT_FIX for now** — the value appears
  in `MatchConfig.min_var_i` defaults and `CompileConfig`, both exposed as public
  configurable fields. Extracting a private constant would save only a few bytes
  of source. Track as technical debt if the threshold needs to vary per-metric.
- Item 7 (OnceLock race): **WONT_FIX** — the race is benign (redundant
  computation, not data corruption) and `OnceLock::get_or_try_init` requires
  Rust 1.80+, above the declared MSRV of 1.70. Revisit when MSRV is raised.
- Item 8 (two-pass template statistics): **WONT_FIX** — template compilation is
  a one-time cost; single-pass savings are negligible compared to the scan cost.
- Item 9 (separability documentation): **WONT_FIX** — `quad2d.rs` is
  `pub(crate)` and the refinement is explicitly labeled as separable 1D+1D. Add
  a doc comment if this module is ever promoted to `lowlevel`.
- Item 10 (AngleGrid float accumulation): **WONT_FIX** — the accumulation loop
  was audited and the off-by-one risk is bounded by the integer cast ceiling.
  An algebraic formula would be equivalent but requires floating-point division
  followed by a ceil, which has the same edge cases. The loop is deterministic
  for any step value that divides the range evenly.
- Item 11 (missing documentation): **Partially addressed** — coordinate system
  documented on `ImageView`. ZNCC/SSD score semantics documented on `Metric`.
  Remaining gaps (precision limits) are WONT_FIX as they are implementation
  detail, not API contract.
- Item 12 (error context loss in parallel): **WONT_FIX** — the angle_idx is
  logged via tracing when the `tracing` feature is enabled. Adding it to the
  error variant would change the public `CorrMatchError` type; deferred to a
  future minor version if demand arises.

## Executive Summary

The corrmatch-rs codebase demonstrates excellent engineering practices:
- Clean trait-based architecture supporting multiple metrics (ZNCC, SSD)
- Thread-safe design with proper synchronization primitives
- Comprehensive error handling using `thiserror`
- Deterministic parallel execution (rayon results match sequential)
- Well-documented public API

One critical bug was identified (rotation casting), along with several high-priority improvements for config validation and API consistency.

---

## Critical Issues

### 1. Undefined Behavior in Rotation Function

**Location:** `src/template/rotate.rs:45-68`

**Issue:** The `rotate_u8_bilinear` function allows small negative floating-point values through its epsilon check, then casts them to `usize`, causing undefined behavior.

```rust
// Line 33-43: Epsilon check allows small negatives
let epsilon = 1e-6;
if src_x < -epsilon || src_y < -epsilon {
    continue;
}
// ...
// Line 47-48: Negative float cast to usize = UB
let x0 = src_x.floor() as usize;  // If src_x = -0.5, floor() = -1.0
let y0 = src_y.floor() as usize;  // Casting -1.0 to usize is UB!
```

**Impact:** Undefined behavior when template pixels map to slightly negative source coordinates.

**Fix:** Add explicit negative check before casting:
```rust
if src_x < 0.0 || src_y < 0.0 {
    continue;
}
let x0 = src_x.floor() as usize;
let y0 = src_y.floor() as usize;
```

---

## High Priority Issues

### 2. No Configuration Validation

**Locations:**
- `src/search/mod.rs:43-92` (MatchConfig)
- `src/bank/mod.rs:24-48` (CompileConfig)

**Issue:** Configuration structs accept invalid values that cause runtime failures or infinite loops:

| Invalid Value | Consequence |
|---------------|-------------|
| `beam_width: 0` | Potential infinite loop in coarse search |
| `per_angle_topk: 0` | Empty results guaranteed |
| `nms_radius: usize::MAX` | All candidates suppressed |
| `coarse_step_deg <= 0.0` | AngleGrid::full() fails |
| `min_step_deg > coarse_step_deg` | Silent misbehavior |

**Recommendation:** Add `validate()` methods that return `CorrMatchResult<()>` and call them from constructors.

### 3. Silent Parallel Degradation

**Location:** `src/search/mod.rs:219-227`

**Issue:** Setting `parallel: true` in MatchConfig without the `rayon` feature silently falls back to sequential execution instead of returning an error.

```rust
#[cfg(feature = "rayon")]
{
    coarse_search_level_par(...)
}
#[cfg(not(feature = "rayon"))]
{
    coarse_search_level(...)  // Silent fallback!
}
```

**Recommendation:** Return `CorrMatchError::ParallelUnavailable` when `parallel: true` but feature is disabled.

### 4. Duplicate API Methods

**Location:** `src/template/plan.rs:111-117`

**Issue:** Both `t_prime()` and `zero_mean()` return the same buffer, creating API confusion.

**Recommendation:** Remove one method or document the intentional alias.

### 5. Inconsistent Rotation Boundary Handling

**Locations:**
- `src/template/rotate.rs:33-43` (unmasked: epsilon tolerance)
- `src/template/rotate.rs:115-120` (masked: strict bounds)

**Issue:** Unmasked rotation uses epsilon tolerance while masked rotation uses strict bounds, creating maintenance risk and potential edge-case differences.

**Recommendation:** Unify boundary checking logic between the two functions.

---

## Medium Priority Issues

### 6. Hardcoded Thresholds

**Locations:** `src/kernel/scalar.rs:66,140,429,499`

**Issue:** The variance threshold `1e-8` is repeated in multiple places without a named constant.

**Recommendation:** Define `const MIN_VARIANCE_THRESHOLD: f32 = 1e-8;` at module level.

### 7. OnceLock Race Condition (Benign)

**Location:** `src/bank/mod.rs:196-245`

**Issue:** Two threads could both pass the `slot.get()` check and compute rotations. While `OnceLock::set()` handles this safely (only one succeeds), it wastes computation.

**Recommendation:** Use `OnceLock::get_or_try_init()` (Rust 1.80+) for exactly-once execution.

### 8. Two-Pass Template Statistics

**Location:** `src/template/plan.rs:28-73`

**Issue:** Template statistics (mean, variance) computed in two passes when single-pass is possible.

**Impact:** Minor performance overhead for small templates.

### 9. Missing Separability Documentation

**Location:** `src/refine/quad2d.rs`

**Issue:** The 2D quadratic refinement applies 1D fits independently (separable assumption). This doesn't work correctly for non-separable peaks but isn't documented.

---

## Minor Issues

### 10. Angle Grid Float Accumulation

**Location:** `src/bank/angles.rs:40-45`

**Issue:** Grid length computed via float accumulation loop instead of algebraic formula, risking off-by-one for certain step values.

```rust
// Current: accumulation
let mut len = 0usize;
loop {
    let angle = min_deg + (len as f32) * step_deg;
    if angle >= max_deg { break; }
    len += 1;
}

// Better: algebraic
let len = ((max_deg - min_deg) / step_deg).ceil() as usize;
```

### 11. Missing Documentation

- Coordinate system (x=column, y=row) not documented at ImageView level
- ZNCC score bounds [-1, 1] documented but precision limits aren't discussed
- SSD negative scores (higher=better) could be confusing without context

### 12. Error Context Loss in Parallel Paths

**Location:** `src/search/coarse.rs:155`

**Issue:** When parallel angle computation fails, the specific `angle_idx` is lost in the error.

---

## Test Coverage Analysis

### Well Tested
- Core ZNCC/SSD kernels with brute-force validation
- Image pyramid construction
- Template plan statistics
- NMS algorithm
- Parallel vs sequential equivalence (ZNCC)

### Coverage Gaps

| Area | Gap |
|------|-----|
| CLI | No integration tests |
| Refinement | Only 2 tests (border + symmetry) |
| Error paths | No tests for invalid configs |
| SSD parallel | Not tested for equivalence |
| Real images | All tests use synthetic patterns |

---

## Architecture Strengths

1. **Trait-based Kernels** - `Kernel` trait allows clean metric abstraction (ZNCC, SSD) with scalar/parallel implementations.

2. **Thread-safe Rotation Cache** - `OnceLock` provides lazy, thread-safe rotation computation.

3. **Comprehensive Error Types** - `CorrMatchError` enum with `thiserror` covers all failure modes with context.

4. **Deterministic Parallelism** - Rayon execution produces identical results to sequential (verified by test).

5. **Clean Public API** - Top-level exports (`Template`, `CompiledTemplate`, `Matcher`, `Match`) hide implementation complexity.

---

## Recommendations by Priority

### Immediate (MVP Blockers)
1. Fix `rotate_u8_bilinear` casting bug
2. Add config validation methods
3. Error on `parallel: true` without rayon feature

### High Priority
1. Create synthetic test validation harness
2. Implement PyO3 bindings for Python users
3. Add CLI integration tests

### Medium Priority
1. Define threshold constants
2. Improve OnceLock usage
3. Consolidate duplicate API methods
4. Add coordinate system documentation

### Low Priority
1. Single-pass template statistics
2. Algebraic angle grid length
3. Extended refinement tests
4. Real-world image test cases

---

## Files Reference

| Category | Key Files |
|----------|-----------|
| Public API | `src/lib.rs` |
| Matching Pipeline | `src/search/mod.rs`, `coarse.rs`, `refine.rs` |
| Kernels | `src/kernel/scalar.rs`, `mod.rs` |
| Template Handling | `src/template/plan.rs`, `rotate.rs` |
| Configuration | `src/search/mod.rs:43-92`, `src/bank/mod.rs:24-48` |
| Error Types | `src/util/error.rs` |
| Tests | `tests/*.rs` |
