# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CorrMatch is a CPU-first template matching library for grayscale images in Rust. It implements coarse-to-fine pyramid search with optional rotation and two metrics: ZNCC (Zero-Mean Normalized Cross-Correlation) and SSD (Sum of Squared Differences). The focus is deterministic, reproducible matching for industrial inspection and document alignment.

## Build Commands

```bash
# Quality gates (run after modifications)
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features

# Build library and CLI
cargo build
cargo build -p corrmatch-cli

# Run CLI
cargo run -p corrmatch-cli -- --config config.json
cargo run -p corrmatch-cli -- --print-schema
cargo run -p corrmatch-cli -- --print-example

# Run specific test
cargo test test_name
cargo test --features rayon  # with parallelism

# Benchmarks
cargo bench

# Documentation
cargo doc --no-deps --workspace
```

## Architecture

### Workspace Structure
- `src/` - Main library crate
- `corrmatch-cli/` - JSON-driven CLI tool
- `tests/` - Integration tests
- `benches/` - Criterion benchmarks
- `tools/synth_cases/` - Python synthetic test case generator

### Core Matching Flow

1. **Template Preparation**: `Template` → `CompiledTemplate` (builds pyramids, optional rotation banks)
2. **Matching**: `Matcher` with `MatchConfig` runs coarse-to-fine search
3. **Coarse-to-Fine**: Exhaustive scan at coarsest level → ROI refinement through finer levels → NMS → subpixel/subangle refinement
4. **Result**: `Match { x, y, angle_deg, score }`

### Key Modules
- `src/bank/` - Compiled template assets and rotation banks (`CompileConfig`, `CompiledTemplate`)
- `src/search/` - Coarse-to-fine matching pipeline (`Matcher`, `MatchConfig`)
- `src/kernel/` - Score evaluation kernels (ZNCC, SSD with scalar/rayon variants)
- `src/image/` - `ImageView` (borrowed 2D view with stride), `ImagePyramid`
- `src/template/` - Template representation, rotation, and plans
- `src/refine/` - Subpixel (2D quadratic) and subangle (1D quadratic) refinement
- `src/lowlevel.rs` - Advanced building blocks for custom pipelines

### Feature Flags
- `rayon` - Parallel search execution (opt-in via `MatchConfig.parallel`)
- `simd` - SIMD-accelerated kernels (planned)
- `image-io` - File I/O via the `image` crate

Default build must work with `--no-default-features`.

## Coding Standards

- Rust 2021 edition, stable toolchain
- Every public item must have rustdoc
- Use `Result<T, CorrMatchError>` with `thiserror`
- No allocations in hot loops unless justified
- No `unsafe` outside `src/kernel/simd.rs`; any unsafe must be documented with invariants
- Parallel search must remain deterministic (results identical to sequential)
