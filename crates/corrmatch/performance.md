# Performance

CorrMatch has two sources of performance data:

1) Criterion microbenchmarks in `crates/corrmatch/benches/corrmatch.rs` (recommended for tracking).
2) Tracing-based pipeline timings (useful during profiling sessions).

## Running Criterion benches

```bash
cargo bench -p corrmatch --bench corrmatch -- --noplot --sample-size 30 --warm-up-time 1 --measurement-time 2
cargo bench -p corrmatch --bench corrmatch --features rayon -- --noplot --sample-size 30 --warm-up-time 1 --measurement-time 2
cargo bench -p corrmatch --bench corrmatch --features simd -- --noplot --sample-size 30 --warm-up-time 1 --measurement-time 2
cargo bench -p corrmatch --bench corrmatch --features "rayon simd" -- --noplot --sample-size 30 --warm-up-time 1 --measurement-time 2
```

## Latest recorded Criterion results (2026-01-31)

Environment:
- OS: Darwin 25.2.0 (arm64)
- CPU: Apple M4 Pro
- Rust: rustc 1.91.0, cargo 1.91.0

All times are Criterion 95% CI `[low mid high]`:

### Default features (none)

- zncc_unmasked_rotation_off: `[10.250 ms 10.313 ms 10.377 ms]`
- ssd_unmasked_rotation_off: `[25.666 ms 25.792 ms 25.933 ms]`
- zncc_masked_rotation_on: `[97.132 ms 97.638 ms 98.060 ms]`
- ssd_masked_rotation_on: `[22.031 ms 22.140 ms 22.232 ms]`

### `--features rayon`

- zncc_unmasked_rotation_off: `[10.140 ms 10.175 ms 10.210 ms]`
- ssd_unmasked_rotation_off: `[25.626 ms 25.713 ms 25.792 ms]`
- zncc_unmasked_rotation_off_parallel: `[5.5567 ms 5.5953 ms 5.6384 ms]`
- zncc_masked_rotation_on: `[98.118 ms 98.567 ms 98.967 ms]`
- ssd_masked_rotation_on: `[22.266 ms 22.302 ms 22.337 ms]`
- zncc_masked_rotation_on_parallel: `[20.017 ms 20.230 ms 20.562 ms]`

### `--features simd`

- zncc_unmasked_rotation_off: `[6.6193 ms 6.6647 ms 6.6957 ms]`
- ssd_unmasked_rotation_off: `[16.344 ms 16.431 ms 16.510 ms]`
- zncc_masked_rotation_on: `[98.821 ms 99.335 ms 99.772 ms]`
- ssd_masked_rotation_on: `[22.268 ms 22.398 ms 22.535 ms]`

### `--features "rayon simd"`

- zncc_unmasked_rotation_off: `[6.5958 ms 6.6248 ms 6.6497 ms]`
- ssd_unmasked_rotation_off: `[16.281 ms 16.368 ms 16.445 ms]`
- zncc_unmasked_rotation_off_parallel: `[3.7322 ms 3.7527 ms 3.7748 ms]`
- zncc_masked_rotation_on: `[98.715 ms 99.098 ms 99.550 ms]`
- ssd_masked_rotation_on: `[22.326 ms 22.429 ms 22.546 ms]`
- zncc_masked_rotation_on_parallel: `[19.914 ms 20.031 ms 20.202 ms]`
