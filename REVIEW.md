# Pre-Release Review — corrmatch-rs
*Reviewed: 2026-04-15*
*Scope: full workspace (`corrmatch`, `corrmatch-cli`, `corrmatch-py`)*

## Review Verdict (2026-04-15, post-implementation)

**Overall: CONDITIONAL PASS — Rust side is release-ready; Python side has a
release-blocking regression in the F3 dataclass layer that the Implementer did
not detect because Python tests were never run locally or in this verification
pass (no `maturin` / `pytest` gate was available in the review sandbox).**

Counts (21 original findings; F22 merged into F21; 3 new from review):

- **verified** (19): F1, F2, F4, F5, F6, F7, F8, F9, F10, F11 (Rust core),
  F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22.
- **needs-rework** (1): F3 — dataclass layer functionally broken at runtime
  (see F23), method-name stub error (see F24), plus minor `Match`-dataclass
  deviation from the triage decision.
- **new-issue-from-review** (3): F23 (blocker), F24 (blocker), F25 (minor).
- **regressions**: 0 in the Rust library. The Python test/viz call sites
  regress only because F3 partially landed without updating them.

Post-review patch (2026-04-15): F23, F24, F25 verified; Python gate green.
F3 Match dataclass (frozen=True, slots=True) implemented in `_wrappers.py`.
Counts update: verified (23): all above + F3, F23, F24, F25.

**Final gate results (2026-04-15, run by parent Opus):**
- `cargo fmt --all -- --check`: clean
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: clean
- `cargo test --workspace --all-features`: all green (incl. 2 doc-tests)
- `maturin develop --release` in `crates/corrmatch-py`: clean
- `pytest crates/corrmatch-py/python/tests -v`: 29 passed, 16 skipped
  (skips are the expected `test_match_absent` cases for match-present cases).
  **`TestMatcherReuse::test_matcher_reusable_after_first_call` (F11) passes.**
- `mypy --strict crates/corrmatch-py/python/corrmatch`: clean.

**Small mypy follow-ups fixed in the parent after the third Implementer
pass (purely stub/annotation work, no runtime impact):**
- `_corrmatch.pyi`: removed duplicate `Template.width` / `Template.height`
  property stubs (re-declared around line 149; the originals at ~125 are
  kept).
- `viz.py:_rot_cw`: bound the `@` result into a typed local to give
  `ndarray` a concrete annotation (was returning `Any`).
- `viz.py`: removed a now-unused `# type: ignore[attr-defined]` on the
  `fig.corrmatch_axes` dict assignment.

**0.2.0 is release-ready on both the Rust and Python sides.**

### Notable concerns raised to the user

1. **(BLOCKER) F3 breaks the existing Python test suite at runtime.** The
   Implementer replaced `corrmatch.MatchConfig` and `corrmatch.CompileConfig`
   at the package top level with frozen dataclasses, but did **not** update
   the Python tests in `test_synthetic.py` / `test_viz.py` or the runtime path
   in `corrmatch/viz.py` to call `.to_rust()` before passing the config into
   PyO3 methods (`compiled.matcher(cfg)`, `tpl.compile(cfg)`). PyO3 extracts
   `Option<MatchConfig>` where `MatchConfig` is the `#[pyclass]`, so passing
   a pure-Python dataclass will raise `TypeError` at runtime. Additionally,
   `test_config_validation` / `test_invalid_metric` / `test_invalid_rotation_mode`
   in `test_synthetic.py` expect exceptions to be raised by
   `corrmatch.MatchConfig(...)` at construction time — but the dataclass has
   no validation and silently accepts `beam_width=0`, `metric="invalid"`,
   etc. See **F23** below for the full write-up. This is why `test_py.yml`
   now includes `mypy --strict`, but mypy will not catch runtime-only
   dataclass-vs-pyclass type mismatches. `pytest` on CI will be red.

2. **`.pyi` stub has a method-name mismatch.** `_corrmatch.pyi` declares
   `Matcher.match_image_topk(...)`, but the actual PyO3-exposed method is
   `match_topk(...)` (`crates/corrmatch-py/src/lib.rs:492`). `mypy --strict`
   of any caller using `matcher.match_topk(...)` — such as
   `viz.py:344` and `test_synthetic.py:166` — will fail with `"Matcher" has
   no attribute "match_topk"`. See **F24**.

3. **Unused imports in the new Python files.** `config.py:15` imports
   `field` from `dataclasses` (never used); `__init__.pyi:8-10` imports
   `Tuple`, `np`, `NDArray` (never used). `mypy --strict` with default
   settings does not flag unused imports, but `ruff` / `flake8` (if ever
   added to CI) would. Trivial cleanup. See **F25**.

4. **`Match` as a dataclass was skipped.** F3 plan item #3 called for
   "`Match` as a dataclass (`@dataclass(frozen=True, slots=True)`)" with
   structured `__eq__`. The Implementer re-exports the raw PyO3 `Match`
   instead. Minor deviation — the PyO3 `Match` has `__repr__`, `get`
   accessors, and would compare equal by Python identity; structured
   equality is not there. Acceptable for 0.2.0 if documented, but the
   triage decision explicitly called for it. Noted as a sub-bullet of F3.

5. **F11 Arc-backed matcher is correct on the Rust side** (`Matcher::from_arc`,
   `Arc<CompiledTemplate>` in `src/search/mod.rs`, and Python `CompiledTemplate`
   using `Arc::clone()` — all verified), but the Python tests that exercise
   it (`test_synthetic.py::test_topk_matches` re-uses `compiled.matcher()`
   implicitly via `match_topk`) will not actually run due to the F3 blocker.
   Once F3 is repaired, F11's idempotency should be tested explicitly with a
   test that calls `compiled.matcher(cfg1)` and `compiled.matcher(cfg2)` on
   the same `CompiledTemplate`. Currently no such test exists.

### How well did the Implementer execute?

The Rust-side work is strong: the workspace restructure (F21) is clean and
correct; `cargo fmt`, `cargo clippy --all-features`, `cargo test --workspace
--all-features`, and `RUSTDOCFLAGS="-D warnings" cargo doc` all pass; version
bumps are consistent; CHANGELOG is well-written; the opaque `CompiledTemplate`
(F2), `#[non_exhaustive]` rollout (F1), `ValidCoord` promotion (F13), MSRV CI
(F4), audit ignore removal (F6), `t_prime` deletion (F8), dead-code deletion
(F9), API alias removal (F10), invariant comments (F17), rotate-asymmetry
docs (F12), `lowlevel` doctest (F16), benchmarks (F19), metadata polish (F20),
and CODE_REVIEW.md cleanup (F18) all match their Resolution lines faithfully.

The Python-side work (F3 + F14 + `.pyi` stubs + mypy gate) is only half-done:
the dataclass layer and stubs exist and read nicely, but the Implementer did
not exercise the new API end-to-end. Running `maturin develop` + `pytest
crates/corrmatch-py/python/tests` locally would have caught both F23 and F24
in under two minutes. Given how well the rest of the review was executed,
this looks like a last-minute "Python tests require maturin, skip for now"
tradeoff that the Implementer did not surface.

**Recommended next step for the user**: repair F23 and F24 before merging
0.2.0. F23 fix options are (a) update tests/viz to call `.to_rust()`, or
(b) accept both dataclass and PyClass in the PyO3 methods via a Python-side
pre-conversion shim. Option (a) is smaller and matches the `to_rust()` API
advertised in the config.py docstrings.

## Triage Decisions (2026-04-15)

User triaged the audit and chose:

1. **Break API now at 0.2.0** — accept all SemVer-breaking design fixes
   (`#[non_exhaustive]`, opaque `CompiledTemplate`, `ValidCoord` → u32,
   config/API cleanup). Bump all three crates to **0.2.0** and add a
   CHANGELOG entry for the release. Better to break once at 0.2.0 than
   regret the surface shape after stabilization.
2. **Hand-written `.pyi` stubs + dataclass-style Python surface + full
   docstrings** for the PyO3 bindings. Upgrade F3 from "ship stubs" to
   "build a Python-native layer" (Python-style snake_case method names
   are already present; add `@dataclass` wrappers where they simplify
   config, full Google-style docstrings, and `.pyi` stubs).
3. **Workspace restructure**: move all crates into `crates/corrmatch`,
   `crates/corrmatch-cli`, `crates/corrmatch-py`. `release-pypi.yml`
   already points to `crates/corrmatch-py/Cargo.toml` (silently broken
   since the directory doesn't exist yet), so this both fixes CI and
   cleans up the top-level layout. Added as **F21**.
4. **F11 (Python `matcher()` semantics)** — go with the Arc-backed
   reuse option: `CompiledTemplate.matcher()` becomes idempotent and
   can be called repeatedly with different configs without recompiling
   the template.
5. **F13 (`ValidCoord`)** — bump to `u32`. Negligible memory impact,
   no truncation risk.
6. **No scope cuts.** Every finding is in scope.

All findings below have had their `Status` and `Fix` sections updated to
reflect these decisions.

**Implementation order for the Implementer agent:**

1. **F21 first** (restructure). This touches every file path and is easier
   to do in one pass with `git mv` before other edits start rewriting file
   contents. Run `cargo test --workspace` after the restructure lands to
   confirm the baseline still works.
2. **F4** (MSRV CI), **F5** (path-dep version pin), **F15** (workspace
   package inheritance). These are purely metadata — cheap to do once the
   crates are in place.
3. **F1, F2, F13** (API breakage for 0.2.0). Do these together since they
   all produce breaking changes and need one CHANGELOG entry.
4. **F8, F10** (duplicate API cleanup), **F9** (dead code removal).
5. **F7, F12** (internal invariants, rotation boundary unification).
6. **F3** (Python-native layer). Biggest isolated task; do after the Rust
   side is stable.
7. **F11** (Arc-backed `matcher()`). Touches Rust core (add
   `Arc<CompiledTemplate>` support) and Python wrapper.
8. **F14** (Python version matrix), **F6** (remove stale audit ignore).
9. **F16, F17, F18, F19, F20** (P3 polish).

## Executive Summary

corrmatch-rs is in excellent shape for a 0.1.x release. The architecture is clean
and well-layered (the high-level `Matcher` / `Template` / `CompiledTemplate` façade
over a `lowlevel` module; feature-gated rayon, simd, image-io, tracing), the
public API is documented, and the test harness covers both unit kernels and a
synthetic ground-truth validation suite. Determinism of the rayon path is
explicitly proven by `tests/rayon_equivalence.rs`, and there is **zero `unsafe`
code** across the workspace — a strong correctness posture. CI covers Linux,
Windows, and macOS with fmt/clippy/test/doc and a separate Python wheel build
job.

The concerns that remain are almost entirely about **API evolvability** and
**release ergonomics**, not correctness:

1. **No `#[non_exhaustive]` anywhere.** `CorrMatchError`, `Metric`, `RotationMode`,
   `CompiledTemplate`, `MatchConfig`, and `CompileConfig` are all exhaustive, so
   any added variant or field after 1.0 is a breaking change. This should be
   decided *before* a stability promise.
2. **`CompiledTemplate` exposes its internal variants and inner structs publicly**,
   which leaks the `Rotated` / `Unrotated` split as part of the API. Matching
   clients can depend on the variants; future refactoring is blocked.
3. **The Python package ships `py.typed` but no `.pyi` stubs**, so tools like
   mypy/pyright advertise-and-find-nothing. This should be fixed or `py.typed`
   removed before a PyPI release.
4. **No MSRV verification in CI.** `rust-version = "1.70"` is declared but not
   enforced.
5. **Several smaller items** around dead code, redundant API, stale audit
   ignore, `CompileConfigNoRot` validation, and documentation gaps for the
   `lowlevel` surface.

None of the findings below are release blockers in the hard sense (no unsafe UB,
no security vulnerabilities, no broken public examples). All are ship-gates for
a *stable* API; most are ship-gates for a polished *PyPI* release.

## Findings

### F1 — No `#[non_exhaustive]` on public enums or structs
- **Severity**: P1
- **Category**: design
- **Location**: `src/util/error.rs:10`, `src/search/mod.rs:28,37,46`, `src/bank/mod.rs:115,521`
- **Status**: verified
- **Review note**: `#[non_exhaustive]` confirmed on `CorrMatchError` (error.rs:9),
  `Metric` (search/mod.rs:28), `RotationMode` (search/mod.rs:38), `MatchConfig`
  (search/mod.rs:48), `CompileConfig` (bank/mod.rs:114), `CompileConfigNoRot`
  (bank/mod.rs:169). `Match` / `Peak` / `ScanParams` correctly left exhaustive.
  CLI and Python bindings updated to `Default::default()` + field mutation.
  Match arms on `Metric` and `RotationMode` in corrmatch-py have `_` wildcards
  (lib.rs:239, 249, 311, 316). Version bumped to 0.2.0 across all crates.
- **Resolution**: Added `#[non_exhaustive]` to `CorrMatchError`, `Metric`, `RotationMode`,
  `MatchConfig`, `CompileConfig`, and `CompileConfigNoRot`. All external call sites updated
  to use `Default::default()` + field mutation pattern (tests, benchmarks, CLI, Python bindings).
  Pattern matches on `Metric` and `RotationMode` in corrmatch-py got `_` wildcard arms.
  Doc-test in `src/lib.rs` updated. Version bumped to 0.2.0 across all crates and
  `pyproject.toml`. CHANGELOG `## 0.2.0` section added.
- **Problem**: `CorrMatchError` (13 variants), `Metric`, `RotationMode`,
  `CompiledTemplate`, `MatchConfig`, and `CompileConfig` are all declared
  exhaustive. Once this crate is published, adding a new error variant, a new
  metric, a new rotation mode, or a new config field is a SemVer-breaking
  change. `grep` confirms zero uses of `#[non_exhaustive]` in the workspace.
  For a library that still anticipates new metrics, new refinement paths, and
  new error surfaces, this is a trap that is easiest to set *before* 1.0.
- **Fix**: Add `#[non_exhaustive]` to:
  - `CorrMatchError` (most important — error enums almost always grow)
  - `Metric` and `RotationMode`
  - `MatchConfig` and `CompileConfig` (struct-level `#[non_exhaustive]` forces
    callers to use the `..default()` or builder pattern, not struct literals)
  Leave `Match`, `Peak`, `ScanParams` exhaustive since their shape is fixed.
  For `CompiledTemplate`, see F2.
  **Version bump**: Bump all three crates (`corrmatch`, `corrmatch-cli`,
  `corrmatch-py`) to `0.2.0` in their respective `Cargo.toml` files and in
  `corrmatch-py/pyproject.toml`. Add a `## 0.2.0 - 2026-04-15` section to
  `CHANGELOG.md` listing the breaking changes from F1, F2, F8, F10, F13.

### F2 — `CompiledTemplate` leaks its rotated/unrotated variants and inner types
- **Severity**: P1
- **Category**: design
- **Location**: `src/bank/mod.rs:521-526`, `src/bank/mod.rs:202,447`
- **Status**: verified
- **Review note**: `CompiledTemplate` is now `pub struct CompiledTemplate { kind:
  CompiledTemplateKind }` with a private `enum CompiledTemplateKind`
  (bank/mod.rs:531-545). `CompiledTemplateRot` and `CompiledTemplateNoRot` are
  `pub(crate)`. `is_rotated()` added (bank/mod.rs:563). The search-time check
  in `src/search/mod.rs:309` correctly uses `!self.compiled.is_rotated()`. No
  public pattern-match points remain.
- **Resolution**: `CompiledTemplate` converted from `pub enum` to `pub struct` wrapping a private
  `enum CompiledTemplateKind`. `CompiledTemplateRot` and `CompiledTemplateNoRot` made `pub(crate)`.
  Added `is_rotated()` public method. The one internal use of `matches!(self.compiled,
  CompiledTemplate::Unrotated(_))` in `search/mod.rs` updated to `!self.compiled.is_rotated()`.
  `CompiledTemplate::compile` alias removed (F10 combined here; use `compile_rotated` directly).
- **Problem**: `CompiledTemplate` is a public enum with public variants
  `Rotated(CompiledTemplateRot)` and `Unrotated(CompiledTemplateNoRot)`, and
  both inner types are public structs. External users can pattern-match on the
  enum (e.g. `search/mod.rs:289` does `matches!(self.compiled, CompiledTemplate::Unrotated(_))`
  — internal use is fine, but the same is possible from outside). This makes
  the rotated/unrotated split part of the stable API surface, blocks a future
  merged representation, and exposes types (`CompiledTemplateRot` etc.) that
  are really implementation details.
- **Fix**: Convert `CompiledTemplate` into an opaque `pub struct` wrapping a
  private `enum CompiledTemplateKind`. Keep `compile_rotated` /
  `compile_unrotated` / `num_levels` / `level_size` as methods. Make
  `CompiledTemplateRot` and `CompiledTemplateNoRot` `pub(crate)` (they are
  referenced internally but need not be in the public API).

### F3 — Build a Python-native surface with dataclasses, docstrings, and `.pyi` stubs
- **Severity**: P1
- **Category**: docs / design (Python API)
- **Location**: `corrmatch-py/python/corrmatch/__init__.py`, `corrmatch-py/src/lib.rs`;
  no `*.pyi` files present
- **Status**: verified
- **Review note**: the infrastructure landed cleanly — `config.py` with typed
  `@dataclass(frozen=True, kw_only=True)` configs, `.to_rust()` / `.from_rust()`
  helpers, Google-style docstrings, `__init__.pyi` and `_corrmatch.pyi` stubs,
  updated `pyproject.toml`, and mypy-strict CI gate. **However** the fix had
  three functional defects that break the Python test suite at runtime (none
  caught because Python tests were not executed during implementation):
  1. **Blocker (F23)**: `corrmatch.MatchConfig` and `corrmatch.CompileConfig`
     at the top level are now the Python dataclasses, but every Python test
     and `viz.py` still passes them directly into PyO3 methods (`tpl.compile
     (cfg)`, `compiled.matcher(cfg)`) that expect the `#[pyclass]` types.
     Every call site should either `.to_rust()` first, or the tests / viz
     should import from `corrmatch._corrmatch` directly.
  2. **Blocker (F24)**: `_corrmatch.pyi` declares `Matcher.match_image_topk`
     but the PyO3 code exposes `match_topk`. `pytest` + `mypy --strict` both
     will fail.
  3. Minor deviation from plan: F3 plan item #3 ("`Match` as a dataclass
     with `@dataclass(frozen=True, slots=True)`") is not implemented — `Match`
     is just re-exported from `_corrmatch`. Acceptable for 0.2.0 but diverges
     from the triage decision.
  4. Minor (F25): unused `field` import in `config.py`; unused `Tuple`, `np`,
     `NDArray` imports in `__init__.pyi`.
  All defects repaired in the post-review patch (see F23, F24, F25 resolutions).
- **Resolution**: Created `crates/corrmatch-py/python/corrmatch/config.py` with
  `@dataclass(frozen=True, kw_only=True)` versions of `CompileConfig` and `MatchConfig`,
  including `.to_rust()` / `.from_rust()` helpers and full Google-style docstrings.
  Updated `__init__.py` to re-export the dataclass versions at the top level while
  keeping the raw `_corrmatch.*` types accessible. Created hand-written stubs:
  `_corrmatch.pyi` (covers all raw PyO3 classes and functions) and `__init__.pyi`
  (re-exports for top-level package). Kept `py.typed` marker (now accurate).
  Mypy `--strict` gate added to `test_py.yml`. Python version matrix expanded to
  include 3.12 (F14 combined here). Post-review: added `_wrappers.py` with Python shim
  classes (`Template`, `CompiledTemplate`, `Matcher`) that accept dataclasses or PyO3
  types transparently, and `Match` as `@dataclass(frozen=True, slots=True)` with
  `from_rust()` classmethod. Added `__post_init__` validators to config dataclasses.
  Added `mypy.ini` at workspace root with `ignore_missing_imports` for optional deps.
- **Problem**: PEP 561 compliance (`py.typed` marker exists, no stubs) is the
  surface symptom; the deeper gap is that the current Python module is a thin
  re-export of PyO3-generated classes. Users get string-typed enums (`metric
  = "zncc"`), no keyword-only dataclass configs, and repr strings instead of
  structured `__eq__`. A high-quality numerical-Python surface would offer
  `@dataclass(frozen=True)` configs, typed `Literal["zncc", "ssd"]` fields,
  and full Google-style docstrings with `Args`, `Returns`, `Raises`, and
  `Example` sections.
- **Fix**: Layer a thin Python module on top of the PyO3 core:
  1. **Split the package**: `corrmatch/__init__.py` re-exports a pure-Python
     wrapper layer; the raw PyO3 extension stays as `corrmatch._corrmatch`
     (already the case — `module-name = "corrmatch._corrmatch"` in pyproject).
  2. **Dataclass configs**: Introduce `corrmatch/config.py` with
     `@dataclass(frozen=True, kw_only=True)` versions of `CompileConfig` and
     `MatchConfig` that accept typed `Literal["zncc", "ssd"]` / `Literal["enabled",
     "disabled"]` fields. Provide `.to_rust()` / `from_rust()` conversion
     helpers. The PyO3-side classes stay available as
     `corrmatch._corrmatch.CompileConfig` for advanced use.
  3. **`Match` as a dataclass**: Wrap the PyO3 `Match` in a
     `@dataclass(frozen=True, slots=True)` with full field docstrings.
  4. **Docstrings**: Every public Python function/class/method gets a
     Google-style docstring (Args, Returns, Raises, Example). Existing
     PyO3 docstrings are a starting point but are terse.
  5. **`.pyi` stubs**: Ship `corrmatch/__init__.pyi` covering the Python-layer
     public API (the dataclasses, `Template`, `CompiledTemplate`, `Matcher`,
     `match_template`, `rotate_u8_bilinear_masked`, `__version__`) and
     `corrmatch/_corrmatch.pyi` covering the raw extension so type checkers
     see through the re-export. Verify with `python -m mypy --strict
     corrmatch-py/python/corrmatch` (or pyright).
  6. **CI gate**: Add a `mypy --strict` (or `pyright`) step to `test_py.yml`
     that type-checks both the stubs and a small example usage snippet.
  7. **Keep `py.typed`** — it is now accurate.
  8. **Update Python tests** (`corrmatch-py/python/tests/test_synthetic.py`,
     `test_viz.py`) to use the new dataclass API so the tests double as
     usage examples.

### F4 — No MSRV verification in CI
- **Severity**: P1
- **Category**: contracts
- **Location**: `Cargo.toml:13` (declares `rust-version = "1.70"`);
  `.github/workflows/ci.yml` uses `stable` only
- **Status**: verified
- **Review note**: `msrv` job confirmed at `.github/workflows/ci.yml:48-64`,
  pinned to toolchain `1.70`, running `cargo check --workspace` and `cargo
  test --workspace` (default features). `rust-version` lifted into
  `[workspace.package]` in root `Cargo.toml:9` and inherited in all three
  crate manifests via `rust-version.workspace = true`.
- **Resolution**: Added `msrv` job to `.github/workflows/ci.yml` using
  `dtolnay/rust-toolchain@master` with toolchain `1.70`. Runs
  `cargo check --workspace` and `cargo test --workspace` (default features only,
  since some deps may require a newer rustc with all-features). `rust-version`
  was lifted into `[workspace.package]` in root `Cargo.toml` as part of F21.
- **Problem**: The declared MSRV is not tested. Any contributor writing code
  that requires features from 1.72+ will not be caught until a user on 1.70
  opens an issue. This is a hidden contract violation waiting to happen.
  Also note: the internal code review mentions `OnceLock::get_or_try_init`
  (stable since 1.80) as a future improvement — a latent MSRV trap.
- **Fix**: Add a `msrv` job to `ci.yml`:
  ```yaml
  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }
      - uses: dtolnay/rust-toolchain@master
        with: { toolchain: "1.70" }
      - run: cargo check --workspace --all-features
      - run: cargo test --workspace  # default features; some deps may gate on newer rustc
  ```
  Verify the chosen MSRV is reachable with current dependencies (criterion 0.8
  and image 0.25 both require >=1.70-1.74; may need to bump MSRV or drop
  `--all-features` in the MSRV job).

### F5 — `corrmatch-py/Cargo.toml` references a path dependency without a version
- **Severity**: P1
- **Category**: workspace
- **Location**: `corrmatch-py/Cargo.toml:18` (post-restructure: `crates/corrmatch-py/Cargo.toml`)
- **Status**: verified
- **Review note**: `crates/corrmatch-py/Cargo.toml:19` now declares
  `corrmatch = { version = "0.2.0", path = "../corrmatch", features = ["image-io", "rayon"] }`
  — version pin + path consistent with CLI crate.
- **Resolution**: Fixed as part of F21. `crates/corrmatch-py/Cargo.toml` now has
  `corrmatch = { version = "0.2.0", path = "../corrmatch", features = ["image-io", "rayon"] }`.
- **Problem**: `corrmatch = { path = ".." }` has no `version =` constraint, while
  `corrmatch-cli/Cargo.toml:22` has `corrmatch = { version = "0.1.0", path = ".." }`.
  `corrmatch-py` has `publish = false` so crates.io will not care, but the
  inconsistency invites a future maintainer to flip `publish` and break the
  publish step. It also masks version drift during local development.
- **Fix**: Add `version = "0.2.0"` to the dependency line in
  `corrmatch-py/Cargo.toml` (same file, new path after restructure), matching
  the CLI crate. Also update `path = ".."` → `path = "../corrmatch"` as part
  of F21.

### F6 — Stale `RUSTSEC-2024-0436` ignore in audit workflow
- **Severity**: P2
- **Category**: security / workspace
- **Location**: `.github/workflows/audit.yml:18`
- **Status**: verified
- **Review note**: `audit.yml` has no `ignore:` key; default `actions-rust-lang/audit@v1`
  invocation with only `createIssues: false`. Confirmed.
- **Resolution**: Removed `ignore: RUSTSEC-2024-0436` from `audit.yml`. The `paste` crate
  is not in the dependency graph, so the ignore was dead.
- **Problem**: `audit.yml` ignores `RUSTSEC-2024-0436` (the `paste` unmaintained
  advisory), but `grep -n "paste"` on `Cargo.lock` returns nothing — the `paste`
  crate is not in our dependency graph. The ignore is dead and obscures future
  real `paste`-related advisories if they reappear.
- **Fix**: Remove the `ignore:` line. If a transitive dep pulls `paste` back in
  later, add the ignore with a dated comment explaining why.

### F7 — `CompileConfigNoRot` has no `validate()` method
- **Severity**: P2
- **Category**: design / code-quality
- **Location**: `src/bank/mod.rs:167-178`
- **Status**: verified
- **Review note**: `CompileConfigNoRot::validate()` added (bank/mod.rs:184-191),
  rejects `max_levels == 0` with `InvalidConfig`. Called at the top of
  `CompiledTemplateNoRot::compile` (bank/mod.rs:470). The Python binding
  `compile_no_rotation` does not directly call `.validate()` but transitively
  hits it via `CompiledTemplateNoRot::compile`, so the behavior reaches the
  Python caller with the same error type. Acceptable.
- **Resolution**: Added `impl CompileConfigNoRot { pub fn validate(&self) }` that rejects
  `max_levels == 0` with `InvalidConfig`. Called at the top of `CompiledTemplateNoRot::compile`.
- **Problem**: `CompileConfig::validate()` exists and is invoked (e.g.
  `corrmatch-py/src/lib.rs:366`), but `CompileConfigNoRot` does not — a caller
  can pass `max_levels: 0` and receive whatever downstream error
  `ImagePyramid::build_u8` produces. Inconsistent with the sibling config and
  with `MatchConfig::validate()`. Not catastrophic (pyramid building will
  error), but the error message will be less precise (`InvalidDimensions`
  instead of `InvalidConfig { reason: "max_levels must be at least 1" }`).
- **Fix**: Add a parallel `impl CompileConfigNoRot { pub fn validate(&self) -> CorrMatchResult<()> }`
  that rejects `max_levels == 0`, and call it from `compile_unrotated` and from
  the Python binding.

### F8 — Redundant `TemplatePlan::t_prime()` vs `zero_mean()`
- **Severity**: P2
- **Category**: design
- **Location**: `src/template/plan.rs:120-127`
- **Status**: verified
- **Review note**: `grep` for `fn t_prime` across the workspace returns zero
  hits. All call sites use `zero_mean()` directly. Note: `valid_t_prime()`
  (a different method on `MaskedTemplatePlan`) still exists and is in active
  use — not affected by this fix.
- **Resolution**: Removed `t_prime()`. All callers (simd.rs, rayon.rs, unmasked_zncc.rs,
  tests.rs) updated to use `zero_mean()` directly.
- **Problem**: `t_prime()` and `zero_mean()` both return `&self.zero_mean` with
  identical doc strings ("Returns the zero-mean template buffer in row-major
  order."). Two names for one accessor is API clutter and invites divergence.
  Flagged in `docs/CODE_REVIEW.md` as still-open.
- **Fix**: Keep `zero_mean()` (the more descriptive name), delete `t_prime()`.
  Grep the workspace — if external callers exist in `corrmatch-py` or tests,
  update them. If keeping both is intentional (backward compatibility after
  publish), mark `t_prime()` as `#[deprecated(note = "use zero_mean")]` and
  document the alias.

### F9 — Dead code: `refine_to_finer_level` kept "for reference"
- **Severity**: P2
- **Category**: code-quality
- **Location**: `src/search/refine.rs:91-93` (`#[allow(dead_code)]` on
  `refine_to_finer_level`)
- **Status**: verified
- **Review note**: `refine_to_finer_level` symbol is no longer defined; only
  `refine_to_finer_level_batch` / `_unmasked` / `_par` / `_unmasked_par` /
  `_unmasked_zncc_integral` exist as live implementations. `grep` for
  `allow(dead_code)` across `crates/` returns a single hit in
  `synthetic_validation.rs` on a test helper field — unrelated.
- **Resolution**: Deleted `refine_to_finer_level` and its `#[allow(dead_code)]` annotation.
  The batch variant (`refine_to_finer_level_batch`) remains as the live implementation.
- **Problem**: The pre-batch per-angle refinement is retained with
  `#[allow(dead_code)]` and a comment saying "kept for reference and potential
  fallback." This is a recipe for rot — it is not exercised by tests, will not
  be caught if it drifts away from the batch version's behavior, and it
  silently inflates the crate's source code.
- **Fix**: Delete it. The git history preserves it if a fallback is ever
  needed. If a fallback really is desirable as live code, promote it behind a
  feature flag and add at least one test that exercises it (currently there is
  none).

### F10 — `CompiledTemplate::compile` labeled "backwards-compatible default" pre-1.0
- **Severity**: P2
- **Category**: design
- **Location**: `src/bank/mod.rs:539-542`
- **Status**: verified
- **Review note**: the `CompiledTemplate::compile` alias is gone;
  `CompiledTemplate` now only exposes `compile_rotated` and
  `compile_unrotated`. `Template::compile` (in `template/mod.rs`) remains as
  the convenience delegator.
- **Resolution**: Removed as part of F2. The opaque struct rewrite dropped the `compile`
  alias. `Template::compile` (delegates to `compile_rotated`) remains as the preferred
  convenience entry point. Noted in CHANGELOG 0.2.0.
- **Problem**: `CompiledTemplate::compile` is a thin alias for `compile_rotated`
  with comment "backwards-compatible default". Before any release there is
  nothing to be backwards-compatible with, and keeping three entry points
  (`Template::compile`, `CompiledTemplate::compile`, `CompiledTemplate::compile_rotated`)
  multiplies the API surface for the same behaviour. `Template::compile`
  (defined in `template/mod.rs`) already delegates to `compile_rotated`.
- **Fix**: Pick one canonical pair (`compile_rotated` / `compile_unrotated`),
  remove `CompiledTemplate::compile`, and let `Template::compile` be the only
  convenience delegator. Document in the README which one to prefer.

### F11 — Python `CompiledTemplate.matcher()` uses awkward "consume-once via take()" pattern
- **Severity**: P2
- **Category**: design (Python API)
- **Location**: `corrmatch-py/src/lib.rs:409-452`
- **Status**: verified (Rust side and Python side)
- **Review note**: Rust core gained `Matcher::from_arc(Arc<CompiledTemplate>)`
  at `src/search/mod.rs:178`, and `Matcher` now stores `Arc<CompiledTemplate>`
  (line 156). Python `CompiledTemplate` stores `Arc<RustCompiledTemplate>`
  (corrmatch-py/src/lib.rs:412) and `matcher()` (line 446) clones the `Arc`
  — idempotent and reusable, as specified. The old `Option<RustMatcher>::take()`
  machinery is removed. However, no Python unit test currently exercises
  "call `compiled.matcher(cfg1)` then `compiled.matcher(cfg2)` on the same
  `CompiledTemplate`" — add one once F23 is fixed.
- **Resolution**: Python `CompiledTemplate` now stores `Arc<RustCompiledTemplate>` instead of
  `Option<RustMatcher>`. `Matcher::from_arc(Arc<CompiledTemplate>)` added to the Rust core
  (`src/search/mod.rs`). `Matcher::new` internally wraps the owned template in an `Arc`.
  Python `matcher()` clones the `Arc` and is now idempotent. The old "consumed" error path
  and `Option<RustMatcher>::take()` machinery removed. Post-review: added
  `test_matcher_reusable_after_first_call` in `test_synthetic.py` (class `TestMatcherReuse`)
  to exercise the Arc-backed idempotency from Python.
- **Problem**: `matcher()` takes `&mut self` and returns `PyRuntimeError("CompiledTemplate already consumed - recompile the template")`
  on second call. This is a surprising side-effect from Python's perspective
  (Python users expect `obj.method()` to be idempotent unless documented
  otherwise) and forces recompilation of the template for every reconfigured
  matcher. The underlying reason — Rust `Matcher` owns the `CompiledTemplate`
  — is an implementation constraint, not a user-visible requirement.
- **Fix**: Store the compiled template behind `Arc<RustCompiledTemplate>` on
  the Python `CompiledTemplate` wrapper. Each call to `.matcher(cfg)` clones
  the `Arc` and constructs a fresh `RustMatcher::new(compiled).with_config(cfg)`.
  `Matcher.match_image` and `match_topk` are already `&self` on the Rust
  side, so no lock is needed. Remove the `Option<RustMatcher>` "take once"
  machinery and the `PyRuntimeError` for re-use. If the underlying
  `RustCompiledTemplate` cannot currently be wrapped in `Arc` (it is moved
  into `RustMatcher::new`), add a `Matcher::from_arc(Arc<CompiledTemplate>)`
  constructor in the core crate, or make `RustMatcher` hold
  `Arc<CompiledTemplate>` internally — that is a ~5-line change in
  `src/search/mod.rs`. Document in the Python docstring that re-configuring
  is cheap.

### F12 — `lowlevel::rotate_u8_bilinear*` exported but boundary handling is asymmetric
- **Severity**: P2
- **Category**: design / docs
- **Location**: `src/template/rotate.rs:36-43` (unmasked: ±1e-6 epsilon tolerance
  then clamp) vs `src/template/rotate.rs:~115-120` (masked: strict bounds)
- **Status**: verified
- **Review note**: both `rotate_u8_bilinear` and `rotate_u8_bilinear_masked`
  now carry explanatory rustdoc describing the boundary treatment. "strict
  bounds", "4-neighbour footprint", and the deliberate asymmetry are called
  out. Code unchanged as documented.
- **Resolution**: Documented the asymmetry in detail on both public rustdoc comments.
  `rotate_u8_bilinear` explains the epsilon-clamp approach and references the masked variant.
  `rotate_u8_bilinear_masked` explains its strict bounds and 4-neighbour footprint check,
  and documents why it intentionally differs from the unmasked path. Code unchanged —
  the asymmetry is intentional for quality (masked path never blends out-of-bounds values).
- **Problem**: Flagged in `docs/CODE_REVIEW.md` and still present. The two
  rotators differ in what they consider "inside the source image" at the sub
  pixel boundary. This divergence is an attractive footgun — callers who
  assume the masked and unmasked paths agree will get edge mismatches. The
  UB from the earlier `f32 → usize` cast has been fixed by the clamp, but
  the behavioural asymmetry remains.
- **Fix**: Factor out a shared `bounds_check_and_clamp(src_x, src_y, max_x, max_y) -> Option<(usize,usize,usize,usize,f32,f32)>`
  and use it from both functions. At minimum, document the asymmetry on both
  public rustdoc comments if it is intentional (the masked variant deliberately
  drops partial footprints to avoid silent quality loss — that rationale
  belongs in the rustdoc).

### F13 — `ValidCoord` fields are `pub u16` — silent truncation risk above 65535
- **Severity**: P2
- **Category**: design
- **Location**: `src/template/plan.rs:7-14,283,445`
- **Status**: verified
- **Review note**: `ValidCoord { x: u32, y: u32 }` at `template/plan.rs:12-17`.
  Both producer sites (plan.rs:281, plan.rs:443) push with `x as u32, y as u32`.
  CHANGELOG 0.2.0 entry documents the breaking change. Kernel consumers read
  via `.x as usize` / `.y as usize` which works identically for u32.
- **Resolution**: Changed `ValidCoord { x: u32, y: u32 }` (from `u16`). Both push sites
  in `plan.rs` updated to `x as u32, y as u32`. Kernel consumers cast to `usize` which
  works identically for u32. Noted as breaking change in CHANGELOG 0.2.0 entry.
- **Problem**: `ValidCoord { x: u16, y: u16 }` is a public struct in
  `lowlevel`. The CHANGELOG notes a past fix "by removing u16 index truncation
  in masked template plans", but this residual u16 struct is still how
  plan.rs:283 and plan.rs:445 push coordinates. For typical templates (<= 512
  px) this is fine, but the invariant "template width < 65536" is nowhere
  enforced or documented, and is easy to forget when extending the code.
- **Fix**: Change `ValidCoord { x: u16, y: u16 }` → `ValidCoord { x: u32, y: u32 }`.
  Update all producers (`plan.rs:283`, `plan.rs:445`) and consumers (any kernel
  code that reads these coords) to use `u32`. Memory impact is ~2× per valid
  coord, negligible versus the pyramid footprint. This is a breaking change
  to the `lowlevel` API — noted in the 0.2.0 changelog entry.

### F14 — `test_py.yml` tests only Python 3.10
- **Severity**: P2
- **Category**: tests
- **Location**: `.github/workflows/test_py.yml:17`
- **Status**: verified
- **Review note**: matrix now `["3.10", "3.12"]` (test_py.yml:17). Mypy
  `--strict` step added (test_py.yml:52-54). New manifest path
  `-m crates/corrmatch-py/Cargo.toml` confirmed.
- **Resolution**: Matrix expanded to `["3.10", "3.12"]` as part of F3 CI update.
  Mypy `--strict` step added to the workflow.
- **Problem**: `python-version: ["3.10"]` is the only version tested. The
  wheel is built with `abi3-py310`, so wheels run on 3.10+, but nothing
  verifies they actually import and behave on 3.11, 3.12, or 3.13 — exactly
  the versions most end users run today. A breakage introduced by a
  Python-side change (e.g. the `numpy` upgrade) could ship undetected.
- **Fix**: Expand the matrix to `["3.10", "3.12"]` at minimum. Add `"3.13"` if
  numpy wheel availability is not a blocker.

### F15 — `corrmatch-cli/Cargo.toml` does not use `workspace.package` inheritance
- **Severity**: P3
- **Category**: workspace
- **Location**: `corrmatch-cli/Cargo.toml:4-7`
- **Status**: verified
- **Review note**: `crates/corrmatch-cli/Cargo.toml:5-8` now inherits
  `edition`, `license`, `repository`, `rust-version` from the workspace.
  `rust-version` lifted into `[workspace.package]` in root `Cargo.toml:9`.
- **Resolution**: Fixed as part of F21. `crates/corrmatch-cli/Cargo.toml` now uses
  `edition.workspace = true`, `license.workspace = true`, `repository.workspace = true`,
  `rust-version.workspace = true`. `rust-version` was also lifted into
  `[workspace.package]` in the root `Cargo.toml`.
- **Problem**: Root `Cargo.toml` defines `[workspace.package]` with edition,
  license, and repository. `corrmatch-py/Cargo.toml` uses `edition.workspace = true`,
  `license.workspace = true`, `repository.workspace = true`. `corrmatch-cli`
  duplicates the strings instead. Inconsistent; drifts over time.
- **Fix**: Replace the literal values in `corrmatch-cli/Cargo.toml` with
  `.workspace = true`. Also consider lifting `rust-version` into
  `[workspace.package]` so every crate inherits the same MSRV (tied to F4).

### F16 — `lowlevel` module lacks usage examples in rustdoc
- **Severity**: P3
- **Category**: docs
- **Location**: `src/lowlevel.rs:1-5`
- **Status**: verified
- **Review note**: `crates/corrmatch/src/lowlevel.rs:8-55` carries a
  self-contained `no_run` doctest demonstrating
  `MaskedTemplatePlan::from_rotated_u8`, `ImageView::new`, and
  `scan_masked_zncc_scalar_full`. The doctest compiles and passes under
  `cargo test --doc`.
- **Resolution**: Added a `no_run` doc-test example to the module-level doc in
  `crates/corrmatch/src/lowlevel.rs`. The example demonstrates building a
  `MaskedTemplatePlan` from pixel data and a binary mask, constructing an
  `ImageView`, calling `scan_masked_zncc_scalar_full`, and iterating over
  the returned `Peak` results.
- **Problem**: The `lowlevel` module doc is one paragraph and has no code
  example. Given it is explicitly the "advanced users" surface (kernels, scan
  functions, plans), at least one worked example — e.g. running
  `scan_masked_zncc_scalar_full` against a manually built `MaskedTemplatePlan`
  — would make the module self-documenting.
- **Fix**: Add a `no_run` doc-test example to `src/lowlevel.rs` showing the
  minimal call sequence: build a `TemplatePlan`, construct a `ScanParams`,
  call a scan function, read the `TopK` result. This also exercises the
  examples in `cargo doc`.

### F17 — Several `.expect()` panic points on invariants that are not documented at the call site
- **Severity**: P3
- **Category**: code-quality
- **Location**: `src/kernel/scalar/*.rs` (13 sites), `src/search/mod.rs:374,455`,
  `src/template/rotate.rs:59,60,77,102,137,138,157`, `src/image/pyramid.rs:77`
- **Status**: verified
- **Review note**: grep for `Invariant:` finds 15 annotation sites across
  `unmasked_zncc.rs` (3), `unmasked_ssd.rs` (2), `masked_zncc.rs` (2),
  `masked_ssd.rs` (2), `rotate.rs` (5), and `pyramid.rs` (1). Matches the
  Resolution line (14) closely — the one extra hit is from an additional
  comment in `rotate.rs`, which is fine.
- **Resolution**: Added `// Invariant:` comments immediately above each `.expect()` call in
  `kernel/scalar/unmasked_zncc.rs` (3 sites), `kernel/scalar/unmasked_ssd.rs` (2 sites),
  `kernel/scalar/masked_zncc.rs` (2 sites), `kernel/scalar/masked_ssd.rs` (2 sites),
  `template/rotate.rs` (4 sites including the small-image early-return), and
  `image/pyramid.rs` (1 site). Each comment explains the loop-bound math that guarantees
  the row index is in range. The `.expect()` messages are kept as-is for panic diagnostics.
- **Problem**: `.expect("row within bounds for scan")` etc. document *what*
  the invariant is, but not *why* it holds. For a contributor reading a
  kernel cold, "the outer loop iterates 0..(h - th + 1) so y + th <= h" is
  non-obvious from inside an `expect`. These are not unsafe, but they are
  panics on invariant violation; a one-line `// invariant:` comment at each
  scan-loop entry point is cheaper insurance than a `debug_assert!`.
- **Fix**: Replace blanket `.expect("row within bounds")` with either (a) a
  short invariant comment above the loop explaining the loop bound math and
  keeping `.expect`, or (b) `.unwrap_or_else(|| unreachable!("row y={y} but image height={h}"))`
  with a descriptive message. No change needed at the API level.

### F18 — `docs/CODE_REVIEW.md` contains unresolved items from 2026-01-11
- **Severity**: P3
- **Category**: docs
- **Location**: `docs/CODE_REVIEW.md` (items 6, 7, 8, 9, 10, 11, 12 under
  "Medium" and "Minor")
- **Status**: verified
- **Review note**: `docs/CODE_REVIEW.md` now carries a 2026-04-15 status
  section cross-referencing REVIEW.md (F7, F8, F10, F12, F13, F17) and
  explicitly marking items 6, 7, 8, 9, 10, 12 as WONT_FIX with rationale.
  REVIEW.md is the authoritative tracker.
- **Resolution**: Updated `docs/CODE_REVIEW.md` with a new 2026-04-15 status
  section. Items closed by REVIEW.md are cross-referenced (F7, F8, F10, F12,
  F13, F17). Items 6, 7, 8, 9, 10, and 12 are explicitly marked WONT_FIX with
  rationale. Item 11 is marked partially addressed. CODE_REVIEW.md is retained
  as a historical record; REVIEW.md is the authoritative tracker.
- **Problem**: The in-repo review doc lists a mix of resolved and unresolved
  items. Resolved ones carry a "Status update" section, but the unresolved
  ones have no tracking — they risk being forgotten. Some (F8, F12 above)
  overlap items in this review; others (hardcoded `1e-8`, two-pass template
  stats, OnceLock race, separability doc) are new.
- **Fix**: Decide which items in CODE_REVIEW.md are committed work vs. closed
  as WONT_FIX, and either link them to REVIEW.md entries or add a terminal
  "Closed as out of scope" bullet per item. Consider deleting CODE_REVIEW.md
  once REVIEW.md is landed to avoid two overlapping trackers.

### F19 — Benchmarks only cover the high-level `Matcher`; no bench for sequential-vs-parallel, SSD vs ZNCC, masked vs unmasked
- **Severity**: P3
- **Category**: tests
- **Location**: `benches/corrmatch.rs`
- **Status**: verified
- **Review note**: `bench_kernel_dispatch` and `bench_parallel_vs_sequential`
  confirmed in `crates/corrmatch/benches/corrmatch.rs:168,250`. Both
  registered in the `criterion_group!` macro at line 301-302.
- **Resolution**: Added two new Criterion benchmark groups to `crates/corrmatch/benches/corrmatch.rs`:
  1. `bench_kernel_dispatch` — runs masked ZNCC scalar (via `scan_masked_zncc_scalar_full`
     directly against a `MaskedTemplatePlan`), unmasked ZNCC via `Matcher` at a single level
     (exercises the integral path when sequential+ZNCC+no-rotation), and unmasked SSD scalar,
     all on a 512×512 image with a 48×48 template.
  2. `bench_parallel_vs_sequential` — runs ZNCC unmasked sequential vs rayon-parallel on a
     1024×1024 image; the parallel bench is guarded with `cfg!(feature = "rayon")`.
  `BenchmarkId` is used in the parallel group for labeled comparison. Both groups compile
  cleanly with and without the `rayon`/`all-features` flags.
- **Problem**: Criterion is wired up and benchmarks the top-level `Matcher`,
  but we cannot spot regressions in kernel-level choices (scalar vs simd
  kernel, rayon on vs off, unmasked ZNCC integral path vs scan) without adding
  targeted benches. This is especially relevant now that `performance.md`
  documents specific characteristics (e.g. masked kernels dominate runtime).
- **Fix**: Add bench groups for (a) kernel dispatch at one pyramid level:
  scalar unmasked ZNCC vs simd vs integral; (b) rayon parallel vs sequential
  on a 1024×1024 image; (c) masked vs unmasked at a matched size. Small
  change, big payoff in PR review signal.

### F20 — README / CHANGELOG / pyproject metadata polish
- **Severity**: P3
- **Category**: docs
- **Location**: `CHANGELOG.md:3-5`, `corrmatch-py/pyproject.toml:13-18`
- **Status**: verified
- **Review note**: `pyproject.toml` classifiers include Python 3.10/3.11/3.12,
  `Scientific/Engineering :: Image Recognition`, `Artificial Intelligence`,
  and audience classifiers. `[project.urls]` carries `Bug Tracker` and
  `Changelog`. `CHANGELOG.md` has `## Unreleased` first with a "No entries
  yet" note, followed by `## 0.2.0 - 2026-04-15` with well-organized
  breaking-change / added sections.
- **Resolution**: `crates/corrmatch-py/pyproject.toml` classifiers expanded to include
  `Python :: 3.10`, `Python :: 3.11`, `Python :: 3.12`, and topic classifiers
  `Topic :: Scientific/Engineering :: Image Recognition` and `Artificial Intelligence`,
  plus audience classifiers. `[project.urls]` gains `"Bug Tracker"` and `Changelog`
  entries. `CHANGELOG.md` reordered so `## Unreleased` appears first (before versioned
  releases) with a "No entries yet" note and `---` dividers for readability.
- **Problem**: Small release-hygiene items:
  - `CHANGELOG.md` has a bullet-only `## Unreleased` section without a date
    marker or a "No entries yet" note — fine now, worth tightening before
    scripted releases.
  - `pyproject.toml` classifiers omit specific Python version classifiers
    (`"Programming Language :: Python :: 3.10"` etc.) and the package
    category (`"Topic :: Scientific/Engineering :: Image Recognition"`),
    which PyPI uses for filtering.
  - No `Issues` URL in `[project.urls]` despite the GitHub repo supporting
    issues.
- **Fix**: Bring `pyproject.toml` classifiers and URLs up to par with the
  `corrmatch` Cargo.toml; clarify the `Unreleased` section policy.

### F21 — Restructure workspace into `crates/` layout
- **Severity**: P1
- **Category**: workspace
- **Location**: repo root (every Cargo.toml, every workflow, README.md,
  CLAUDE.md, tests that use `CARGO_MANIFEST_DIR`)
- **Status**: verified
- **Review note**: full restructure confirmed. Root `Cargo.toml` is
  workspace-only with `[workspace.package]` holding `edition`, `license`,
  `repository`, `rust-version`. `crates/corrmatch/Cargo.toml` exists with
  version 0.2.0, workspace inheritance, and a correct `include = [...]`
  list. `crates/corrmatch-cli/Cargo.toml` and `crates/corrmatch-py/Cargo.toml`
  both at 0.2.0 with `corrmatch = { version = "0.2.0", path = "../corrmatch", ... }`.
  `test_py.yml` uses `-m crates/corrmatch-py/Cargo.toml` and
  `working-directory: crates/corrmatch-py`. `synthetic_validation.rs:181`
  resolves the synthetic-cases dir via `"../../synthetic_cases"`.
  `conftest.py:12` walks up with `parents[4]` — correct for the new depth.
  Ancillary files (README.md, CHANGELOG.md, ROADMAP.md, performance.md,
  LICENSE) are copied into `crates/corrmatch/`. `cargo build --workspace`,
  `cargo test --workspace --all-features`, `cargo doc --all-features` all
  pass.
- **Resolution**: `git mv` step done by parent agent. Root `Cargo.toml` rewritten
  as workspace-only manifest with `[workspace.package]` including `rust-version`.
  `crates/corrmatch/Cargo.toml` created with package metadata, version 0.2.0, and
  workspace inheritance. `crates/corrmatch-cli/Cargo.toml` and
  `crates/corrmatch-py/Cargo.toml` updated: versions bumped to 0.2.0, paths fixed
  to `../corrmatch`, `corrmatch-py` gains `rust-version.workspace = true`.
  `pyproject.toml` version bumped to 0.2.0. `test_py.yml` updated with new manifest
  path and working-directory. `synthetic_validation.rs` path fixed to
  `../../synthetic_cases`. `conftest.py` walk-up count fixed from `parents[3]` to
  `parents[4]`. CLAUDE.md and AGENTS.md updated with new paths. Ancillary files
  (README.md, CHANGELOG.md, ROADMAP.md, performance.md, LICENSE) copied into
  `crates/corrmatch/` for `cargo package`. `cargo build --workspace` and
  `cargo test --workspace` both pass clean.
- **Problem**: The current layout mixes the root `corrmatch` library
  (Cargo.toml, src/, tests/, benches/) with sibling crates (`corrmatch-cli/`,
  `corrmatch-py/`) at workspace root. This is functional but has two real
  costs: (1) the root `Cargo.toml` is both a package *and* the workspace
  manifest, so top-level conventions about "what is workspace vs what is
  package" are blurred; (2) `.github/workflows/release-pypi.yml:31,56`
  already references `crates/corrmatch-py/Cargo.toml`, which does not
  exist — the PyPI release workflow is silently broken until the restructure
  happens (or the paths are reverted, which we are not doing).
- **Fix**: Move crates under a `crates/` directory:
  ```
  corrmatch-rs/
    Cargo.toml            # workspace-only (no [package])
    CHANGELOG.md, README.md, ROADMAP.md, LICENSE, AGENTS.md, CLAUDE.md
    synthetic_cases/      # stays at root (loaded via CARGO_MANIFEST_DIR+/..)
    tools/                # stays at root
    book/                 # stays at root
    docs/                 # stays at root
    rustfmt.toml          # stays at root
    crates/
      corrmatch/
        Cargo.toml        # the library package
        src/
        tests/
        benches/
        README.md         # copy or symlink of root README (needed for cargo package)
        CHANGELOG.md      # copy or symlink
        LICENSE           # copy or symlink
      corrmatch-cli/
        Cargo.toml        # corrmatch dep: path = "../corrmatch", version = "0.2.0"
        src/
        config.example.json, config.schema.json
      corrmatch-py/
        Cargo.toml        # corrmatch dep: path = "../corrmatch", version = "0.2.0"
        src/
        python/
        pyproject.toml
        build.rs
        README.md
  ```
  Concrete tasks:
  1. **Move directories**: `git mv` the contents of each crate into its
     `crates/<name>/` target. Use `git mv` (not `cp`) to preserve history.
     `src/`, `tests/`, `benches/` from root move into `crates/corrmatch/`.
  2. **Split root Cargo.toml**: remove the `[package]`, `[dependencies]`,
     `[dev-dependencies]`, `[features]`, `[[bench]]`, and `include` sections
     from the root file; keep only `[workspace]` (update members to the
     new paths) and `[workspace.package]`. Create
     `crates/corrmatch/Cargo.toml` holding the package metadata that used
     to live at root, using `edition.workspace = true`,
     `license.workspace = true`, `repository.workspace = true`,
     `rust-version.workspace = true` (ties to F4).
  3. **Update path dependencies**:
     - `crates/corrmatch-cli/Cargo.toml`: `corrmatch = { version = "0.2.0", path = "../corrmatch", features = ["image-io", "tracing"] }`
     - `crates/corrmatch-py/Cargo.toml`: `corrmatch = { version = "0.2.0", path = "../corrmatch", features = ["image-io", "rayon"] }`
  4. **Update CI workflow paths**:
     - `.github/workflows/test_py.yml:41`: `-m corrmatch-py/Cargo.toml` → `-m crates/corrmatch-py/Cargo.toml`
     - `.github/workflows/test_py.yml:49`: `working-directory: corrmatch-py` → `working-directory: crates/corrmatch-py`
     - `.github/workflows/release-pypi.yml` already uses `crates/corrmatch-py/Cargo.toml` — no change needed (this confirms the direction).
     - `.github/workflows/ci.yml`, `release.yml`, `audit.yml`, `publish-docs.yml`: no path changes (they all use workspace-level commands).
  5. **Fix `synthetic_cases` discovery** from tests:
     - `tests/synthetic_validation.rs:181` builds the path as
       `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("synthetic_cases")`.
       After the move, `CARGO_MANIFEST_DIR` is `crates/corrmatch/`, so
       change to `.join("../../synthetic_cases")` (or use an env var
       set in `.cargo/config.toml`).
     - `corrmatch-py/python/tests/conftest.py:13` computes
       `REPO_ROOT / "synthetic_cases"` — verify `REPO_ROOT` is still the
       repo root after the move (likely yes, since conftest walks up from
       the test file). Adjust the walk-up count if needed.
  6. **Fix include lists** (for `cargo package`): ensure
     `crates/corrmatch/Cargo.toml`'s `include = [...]` references files
     *inside* the crate directory. Copy or symlink `README.md`, `CHANGELOG.md`,
     `ROADMAP.md`, `performance.md`, `LICENSE` into `crates/corrmatch/`
     (cargo requires these to live inside the package to be packaged).
     Prefer symlinks on Unix — Windows CI will see the target file.
     Alternative: drop them from include and have cargo pick up the
     auto-README from the crate dir.
  7. **Update doc references**:
     - `CLAUDE.md:16-22` (workspace structure section)
     - `AGENTS.md:21` (references `src/kernel/simd.rs` — adjust to
       `crates/corrmatch/src/kernel/simd.rs`)
     - `README.md:185,191,197`: CLI invocation examples use
       `cargo run --manifest-path ../../Cargo.toml -p corrmatch-cli`;
       this still works because `cargo -p` resolves workspace members by
       name, not path. Verify and update if needed.
     - `docs/RELEASE_CHECKLIST.md` paths to Cargo.toml files.
     - `tools/viz_case.sh`: verify paths still resolve.
  8. **Verify**: `cargo build --workspace`, `cargo test --workspace --all-features`,
     `maturin build -m crates/corrmatch-py/Cargo.toml`.
  9. **Update `benches/corrmatch.rs`** reference in root Cargo.toml
     `[[bench]]` — this section migrates into `crates/corrmatch/Cargo.toml`
     and the `name = "corrmatch"` still resolves via `crates/corrmatch/benches/corrmatch.rs`.

### F22 — (merged into F21)
- **Severity**: P1
- **Category**: workspace
- **Status**: verified — subsumed by F21
- **Problem**: `release-pypi.yml` currently references `crates/corrmatch-py/Cargo.toml`
  which does not exist, so tag pushes will fail at the wheel-build step. Not a
  separate finding after F21 lands — the restructure creates the directory.
- **Fix**: No additional action; verify by running `gh workflow run release-pypi.yml`
  after F21 completes (or cutting a dry-run tag).

### F23 — Python test suite and `viz.py` pass dataclass configs into PyO3 methods (runtime TypeError)
- **Severity**: P1 (release blocker for PyPI 0.2.0 wheel)
- **Category**: Python API (new-issue-from-review)
- **Location**: `crates/corrmatch-py/python/corrmatch/viz.py:338,343,467`,
  `crates/corrmatch-py/python/tests/test_synthetic.py:50,63,173-189,211,218`,
  `crates/corrmatch-py/python/tests/test_viz.py:10,15,42`
- **Status**: verified
- **Resolution**: Created `_wrappers.py` with thin Python shim classes (`Template`,
  `CompiledTemplate`, `Matcher`, `Match`) that accept either Python-native config
  dataclasses or raw PyO3 types. `Template.compile()` and `CompiledTemplate.matcher()`
  now call `_to_rust_compile()` / `_to_rust_match()` internally, transparently coercing
  dataclasses to PyO3 types. `__init__.py` updated to export shim classes instead of raw
  PyO3 types. `match_template()` wrapped to return Python `Match` dataclass. Added
  `__post_init__` validators to both `CompileConfig` and `MatchConfig` dataclasses that
  call `.to_rust()` at construction time, restoring runtime validation (`beam_width=0`
  raises `RuntimeError`, `metric="invalid"` raises `ValueError`). Added missing return
  type annotations in `viz.py` (`show_matches`, `match_and_show`, `show_from_json`).
- **Problem**: After F3, `corrmatch.MatchConfig` and `corrmatch.CompileConfig`
  at the top-level package namespace are the frozen Python dataclasses from
  `corrmatch.config`. Every remaining call site in `viz.py` and the pytest
  suite still passes these dataclass instances directly into PyO3 methods
  that expect the `#[pyclass]` types:
  - `tpl.compile(corrmatch.CompileConfig(...))` — PyO3 extracts
    `Option<CompileConfig>` with `extract::<CompileConfig>()` which rejects
    anything that isn't an instance of the `#[pyclass]`.
  - `compiled.matcher(corrmatch.MatchConfig(...))` — same problem.
  - `corrmatch.MatchConfig(beam_width=0)` — pre-F3 the PyO3 constructor raised
    `RuntimeError` on validate. The dataclass has no `__post_init__` validator,
    so this silently constructs.
  - `corrmatch.MatchConfig(metric="invalid")` — pre-F3 raised `ValueError`
    ("metric must be 'zncc' or 'ssd'"). `Literal["zncc","ssd"]` is a
    static-typing hint only; at runtime Python accepts any string.
  The net result: `test_synthetic.py::test_match_present`,
  `test_topk_matches`, `test_rotated_template_detection`,
  `test_parallel_matches_sequential`, `test_config_validation`,
  `test_invalid_metric`, `test_invalid_rotation_mode`, and
  `test_viz.py::test_viz_smoke_no_show` will all fail at runtime.
- **Fix options**:
  1. **Call `.to_rust()` at every passthrough.** Update `viz.py:338,343,467`
     and every test call site to e.g.
     `compiled.matcher(match_cfg.to_rust())` and
     `tpl.compile(compile_cfg.to_rust())`. Update the validation tests to
     either instantiate `corrmatch._corrmatch.MatchConfig(...)` directly (the
     raw PyO3 class, which still validates) or to call `.to_rust()` and
     expect the exception on that call. This is the smallest change and
     matches the `.to_rust()` API that `config.py`'s own docstrings already
     advertise.
  2. **Auto-convert inside PyO3.** Teach `CompiledTemplate.matcher`,
     `Template.compile`, and `Template.compile_no_rotation` to accept
     "anything with a `to_rust()` method" via a Python-side shim in
     `__init__.py`. Cleaner long-term but introduces a maintenance burden
     (every new PyO3 method has to be wrapped).
  3. **Add runtime validation to the dataclasses.** Give
     `corrmatch.config.MatchConfig` / `CompileConfig` a `__post_init__` that
     constructs the raw PyO3 equivalent (which runs validation) and
     optionally stores it as a cached attribute. This recovers the
     "construction-time validation" behavior that the pre-F3 tests rely on
     without requiring test changes for the validation assertions, but it
     couples the dataclass to the PyO3 module at construction time (which
     is arguably the right thing).
  Recommended: (1) + (3), so the dataclass is a proper Python-facing
  validator and `.to_rust()` is the one-line idiom for passing it down.

### F24 — `_corrmatch.pyi` declares `Matcher.match_image_topk` but PyO3 exposes `match_topk`
- **Severity**: P1 (release blocker; the new mypy-strict CI gate will fail)
- **Category**: Python API / type stubs (new-issue-from-review)
- **Location**: `crates/corrmatch-py/python/corrmatch/_corrmatch.pyi:194-209`
  (declares `match_image_topk`); `crates/corrmatch-py/src/lib.rs:492` (PyO3
  exposes the name `match_topk`)
- **Status**: verified
- **Resolution**: Renamed `match_image_topk` to `match_topk` in `_corrmatch.pyi`.
  The `_wrappers.Matcher.match_topk()` shim also uses the correct name. All call sites
  in `viz.py` and `test_synthetic.py` already used `match_topk` and now type-check
  correctly.
- **Problem**: method-name mismatch between the stub and the actual Rust
  attribute. Any caller of `matcher.match_topk(image, k=...)` —
  `viz.py:344`, `test_synthetic.py:166` — will type-check as "no such
  attribute" under `mypy --strict`. Conversely, if anyone follows the stub
  and calls `matcher.match_image_topk(...)`, they will get
  `AttributeError` at runtime.
- **Fix**: Rename `match_image_topk` to `match_topk` in `_corrmatch.pyi`,
  *or* add a `#[pyo3(name = "match_image_topk")]` attribute to the PyO3
  `match_topk` method in `src/lib.rs:492` to flip it to the longer,
  Match-suite-consistent name (`match_image_topk` pairs better with the
  existing `match_image`). The Rust-core method is `match_image_topk`, so
  matching that in the Python API is the more consistent choice.

### F25 — Unused imports in the new Python files
- **Severity**: P3
- **Category**: Python code quality (new-issue-from-review)
- **Location**:
  - `crates/corrmatch-py/python/corrmatch/config.py:15` — `from dataclasses
    import dataclass, field` (`field` never used).
  - `crates/corrmatch-py/python/corrmatch/__init__.pyi:8-10` — `Tuple`,
    `np`, `NDArray` imports never referenced in the stub body.
- **Status**: verified
- **Resolution**: Removed unused `field` from `config.py` import. Rewrote `__init__.pyi`
  to match the new shim-based structure; it now imports `NDArray` and `np` (both used
  in the `match_template` stub signature) and no longer imports `Tuple` (not needed).
- **Problem**: Dead imports accumulate lint noise. `mypy --strict` with
  default options does not flag them, but `ruff` / `flake8` (often added
  to PyPI release pipelines) would. Trivial cleanup.
- **Fix**: Drop `field` from the `config.py` import; drop `Tuple`, `np`,
  `NDArray` from `__init__.pyi` (or reference them — but since the stub
  only re-exports from other modules, they are not needed).

## Out-of-Scope Pointers

- The `refine` module's quadratic subpixel/subangle fit and its
  separable-2D assumption (flagged in `docs/CODE_REVIEW.md` item 9) belong to
  **algo-review** / **calibration-review**, not this audit.
- Scalar kernel hot-loop performance (kernel dispatch, unroll, SIMD viability)
  belongs to **perf-architect**. The `performance.md` notes an open question
  about the masked-kernel bottleneck — a targeted perf session would answer it.
- The determinism proof for the rayon path relies on `TopK`'s deterministic
  tie-break (`peak_cmp_desc`). A formal argument for the same property on the
  `par_iter` precomputation in `bank/mod.rs` (rotation slots) would be an
  **algo-review** follow-up.

## Strong Points

- **Zero `unsafe` code across the workspace.** Hard to overstate how much
  correctness work this saves.
- **Deterministic parallelism is explicitly tested** (`tests/rayon_equivalence.rs`)
  — rare and valuable.
- **Feature flags are exemplary**: default build works with `--no-default-features`;
  `tracing` compiles to no-op macros (`src/trace.rs`); optional deps are
  gated with `dep:` syntax; feature-gated code is correctly `#[cfg]`-ed.
- **Error enum is rich and actionable** — 13 variants with structured context,
  not a generic "failed" catchall.
- **Module layout reads cleanly**: one-sentence purpose per module, no
  `utils` / `common` / `helpers` dumping grounds, `lowlevel` correctly
  isolates expert surface from the primary API.
- **Synthetic validation suite** — Rust and Python both validate against
  ground-truth synthetic cases, a level of rigor most 0.1.x crates skip.
- **Config validation is present and invoked** (`MatchConfig::validate`,
  `CompileConfig::validate`) — with the one gap called out in F7.
- **README example compiles against the current API.** Verified that every
  symbol in the README quickstart (`Template`, `CompiledTemplate`, `Matcher`,
  `MatchConfig`, `ImageView`, `RotationMode`, `CompileConfig`) exists with
  the signatures shown.
