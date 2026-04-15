"""CorrMatch - CPU-first template matching library for grayscale images.

This module provides Python bindings for the corrmatch Rust library,
implementing coarse-to-fine pyramid search with optional rotation and
two metrics: ZNCC and SSD.

The recommended API uses the Python-native dataclass configs from
:mod:`corrmatch.config`. The raw PyO3-generated classes are available
under ``corrmatch._corrmatch`` for advanced use.

Example:
    Simple one-shot matching::

        import numpy as np
        import corrmatch

        image = np.zeros((200, 300), dtype=np.uint8)
        template = np.zeros((40, 60), dtype=np.uint8)
        result = corrmatch.match_template(image, template)
        print(f"Found at ({result.x}, {result.y}) score={result.score:.4f}")

    Repeated matching with the same template (more efficient)::

        from corrmatch import Template
        from corrmatch.config import CompileConfig, MatchConfig

        tpl = Template(template)

        # Compile with rotation support (accepts Python dataclass directly)
        compiled = tpl.compile(CompileConfig(max_levels=4, coarse_step_deg=10.0))

        # Create matcher and run (accepts Python dataclass directly)
        matcher = compiled.matcher(MatchConfig(rotation="enabled", beam_width=8))
        result = matcher.match_image(image)

Note:
    ``Template``, ``CompiledTemplate``, and ``Matcher`` in this namespace are
    thin Python shim classes that accept either the Python-native config
    dataclasses (``CompileConfig``, ``MatchConfig``) or the raw PyO3 types.
    The underlying Rust-backed classes are still importable as
    ``corrmatch._corrmatch.Template`` etc.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _corrmatch as _c

# Python shim wrappers that accept dataclasses or PyO3 config types
from ._wrappers import (
    Match,
    Template,
    CompiledTemplate,
    Matcher,
)

# Re-export Python-native config dataclasses under the top-level namespace
# for convenient access.
from .config import CompileConfig, MatchConfig

# Also re-export the raw Rust configs under their own names for callers
# who want direct access without going through config.py.
from ._corrmatch import (
    CompileConfig as _RustCompileConfig,
    MatchConfig as _RustMatchConfig,
    rotate_u8_bilinear_masked,
    __version__,
)


def match_template(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    metric: str = "zncc",
    rotation: str = "disabled",
    parallel: bool = False,
) -> Match:
    """One-shot template matching.

    Compiles the template and finds the best match in a single call.
    Use the ``Template`` / ``CompiledTemplate`` / ``Matcher`` pipeline
    for repeated matching with the same template.

    Args:
        image: 2D uint8 numpy array (height × width).
        template: 2D uint8 numpy array (height × width).
        metric: ``"zncc"`` or ``"ssd"`` (default: ``"zncc"``).
        rotation: ``"enabled"`` or ``"disabled"`` (default: ``"disabled"``).
        parallel: Enable parallel execution (default: False).

    Returns:
        Best ``Match`` result.

    Raises:
        RuntimeError: If matching fails.
        ValueError: If the metric or rotation string is invalid.
    """
    return Match.from_rust(
        _c.match_template(image, template, metric=metric, rotation=rotation, parallel=parallel)
    )


__all__ = [
    # Core types
    "Match",
    "Template",
    "CompiledTemplate",
    "Matcher",
    # Python-native config dataclasses
    "CompileConfig",
    "MatchConfig",
    # Functions
    "match_template",
    "rotate_u8_bilinear_masked",
    # Version
    "__version__",
]
