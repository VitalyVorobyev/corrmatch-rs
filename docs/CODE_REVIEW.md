# CorrMatch Code Review

**Date:** 2026-01-11
**Scope:** Full library review with focus on correctness, design, and structure
**Overall Assessment:** Production-quality with minor issues to address

---

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
