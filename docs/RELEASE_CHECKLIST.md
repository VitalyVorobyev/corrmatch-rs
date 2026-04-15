# Release checklist (crates.io + PyPI)

This is a practical checklist for cutting a tagged release and publishing both:
- the Rust crate `corrmatch` to crates.io
- the Python package `corrmatch` (from `crates/corrmatch-py/`) to PyPI

## 1) Prep

- Bump versions in all three crates and the Python project file:
  - `crates/corrmatch/Cargo.toml`
  - `crates/corrmatch-cli/Cargo.toml`
  - `crates/corrmatch-py/Cargo.toml`
  - `crates/corrmatch-py/pyproject.toml`
- Update `CHANGELOG.md` for the release version/date.
- Run quality gates:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

- Re-run validation + benches:
  - Synthetic validation: `docs/VALIDATION.md`
  - Benches: `performance.md`

## 2) crates.io (Rust)

Sanity checks:

```bash
cargo package -p corrmatch --list
```

Publish order:
1. Publish `corrmatch` first.
2. Publish `corrmatch-cli` (it depends on `corrmatch`).

Commands (interactive login):

```bash
cargo publish -p corrmatch
# wait for crates.io index to update
cargo publish -p corrmatch-cli
```

Notes:
- `corrmatch-cli` should specify both `version = "x.y.z"` and `path = ".."` for `corrmatch`
  while developing; Cargo will strip the `path` on publish.
- `cargo package -p corrmatch-cli` will fail until the referenced `corrmatch` version exists on
  crates.io. This is expected; publish `corrmatch` first.

## 3) PyPI (Python)

Local wheel build + tests:

```bash
python -m pip install -U maturin pytest numpy pillow
maturin build --release -m crates/corrmatch-py/Cargo.toml --out dist
python -m pip install dist/*.whl
python -m pytest crates/corrmatch-py/python/tests
```

Publishing (one option):

```bash
maturin publish --release -m crates/corrmatch-py/Cargo.toml
```

Notes:
- For real releases, prefer CI that builds wheels for Linux/Windows/macOS and uploads them to PyPI.
- Ensure the Python package version matches the Rust crate version.

## 4) Tag and GitHub release

- Tag `vX.Y.Z` and push the tag.
- Verify CI status (Rust + Python).
- Create/update GitHub Release notes and attach any artifacts you want to distribute.

