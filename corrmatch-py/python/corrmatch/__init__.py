"""CorrMatch - CPU-first template matching library for grayscale images.

This module provides Python bindings for the corrmatch Rust library,
implementing coarse-to-fine pyramid search with optional rotation and
two metrics: ZNCC and SSD.

Example usage:

    import numpy as np
    import corrmatch

    # Simple one-shot matching
    image = np.array(...)  # 2D uint8 array
    template = np.array(...)  # 2D uint8 array
    result = corrmatch.match_template(image, template)
    print(f"Found at ({result.x}, {result.y}) with score {result.score}")

    # For repeated matching with the same template (more efficient):
    tpl = corrmatch.Template(template)
    compiled = tpl.compile()  # Or tpl.compile_no_rotation() for faster path
    matcher = compiled.matcher()
    result = matcher.match_image(image)

    # With rotation enabled:
    compiled = tpl.compile(corrmatch.CompileConfig(coarse_step_deg=10.0))
    matcher = compiled.matcher(corrmatch.MatchConfig(rotation="enabled"))
    result = matcher.match_image(image)
"""

from ._corrmatch import (
    Match,
    CompileConfig,
    MatchConfig,
    Template,
    CompiledTemplate,
    Matcher,
    match_template,
    __version__,
)

__all__ = [
    "Match",
    "CompileConfig",
    "MatchConfig",
    "Template",
    "CompiledTemplate",
    "Matcher",
    "match_template",
    "__version__",
]
