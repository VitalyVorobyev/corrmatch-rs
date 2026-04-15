# Synthetic validation

CorrMatch ships a synthetic validation suite under `synthetic_cases/` plus an
integration test runner (`crates/corrmatch/tests/synthetic_validation.rs`).
The goal is to validate end-to-end correctness (coarse search + refinement)
against known ground truth.

## How to run

```bash
cargo test -p corrmatch --test synthetic_validation --all-features -- --nocapture
```

Notes:
- Requires feature `image-io` (enabled by `--all-features`) to load PNGs.
- The test reads `synthetic_cases/manifest.json` and validates all listed cases.

## Acceptance criteria (current)

- Position tolerance: ±3 px
- Angle tolerance: ±2°
- For ZNCC: minimum score threshold 0.80 (0.77 for `occluded_25pct`)
- For SSD: score is negative SSE (higher is better); acceptance is based on
  position/angle tolerances.

## Latest recorded run (2026-01-31)

Environment:
- OS: Darwin 25.2.0 (arm64)
- CPU: Apple M4 Pro
- Rust: rustc 1.91.0, cargo 1.91.0
- Python: 3.14.2

Command:

```bash
cargo test -p corrmatch --test synthetic_validation --all-features -- --nocapture
```

Result:

```text
Passed: 15/15
Failed: 0/15
```

Cases (all PASS):
- clean_translation
- clean_translation_large
- rotation_coarse_30deg
- rotation_fine_22_5deg
- rotation_wrap_172_5deg
- noise_gaussian
- blur_sigma_1_5
- illumination_shift
- occluded_25pct
- distractors_topk
- near_border
- negative_no_match
- pyramid_stress
- pyramid_stress_single_level
- rotation_fine_single_level

