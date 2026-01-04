# CHANGELOG

## Unreleased
- Add core error types and result alias for fallible APIs.
- Add `ImageView`, `ImagePyramid`, `Template`, and `TemplatePlan` foundations.
- Add unit tests for image views, pyramid downsampling, and template stats.
- Add `AngleGrid` and `CompiledTemplate` with lazy rotation caching.
- Add deterministic bilinear rotation for `u8` templates and math helpers.
- Add tests for angle grids, rotations, and cache behavior.
- Add masked rotation plans for ZNCC and scalar masked scan with Top-K support.
- Add spatial NMS helper and masked ZNCC correctness tests.
- Add coarse-to-fine matcher with joint angle search and ROI refinement.
- Add scan helpers for full-range and ROI masked ZNCC evaluation.
- Add end-to-end pipeline tests and angle-step scheduling coverage.
