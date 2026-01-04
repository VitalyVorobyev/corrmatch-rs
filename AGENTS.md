# AGENTS.md

## Project purpose
CorrMatch is a Rust library for CPU-first template matching with ZNCC scoring.
It targets hierarchical search across image pyramids and rotation space.
The focus is deterministic, reproducible matching for vision pipelines.
Primary use cases include industrial inspection and document alignment.
The crate prioritizes correctness, predictable performance, and minimal deps.
Initial releases emphasize scalar kernels and clean APIs for extension.

## Non-goals
- GPU backends or CUDA/OpenCL integrations in v1.
- OpenCV dependency or bindings.
- End-to-end feature detection or keypoint pipelines.

## Coding standards
- Rust 2021, stable toolchain.
- Every public item must have rustdoc.
- Prefer `Result<T, CorrMatchError>` with `thiserror`.
- No allocations in hot loops unless justified.
- No unsafe outside `src/kernel/simd.rs`; any unsafe must be documented with invariants and tested.

## Testing standards
- Unit tests for math/refinement correctness.
- Property-ish tests with random synthetic images (rand) for invariants.
- Add regression tests when bugs are found.

## Benchmark standards
- Criterion benches live in `benches/`, must be reproducible and documented.

## Feature policy
- Default build must work with `--no-default-features`.
- `rayon` and `simd` must not change results beyond documented floating tolerance.
- Parallel search is opt-in via `MatchConfig.parallel` and must remain deterministic.

## Documentation policy
- README shows quickstart + feature flags.
- ROADMAP kept updated as milestones land.

## Build, test, and quality gates

Always run after modifications:

* `cargo fmt --all`
* `cargo clippy --workspace --all-targets --all-features -- -D warnings`
* `cargo test --workspace --all-features`
