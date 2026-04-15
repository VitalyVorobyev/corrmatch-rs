"""Python shim classes that wrap the PyO3 extension types.

These thin wrappers accept either Python-native config dataclasses
(``CompileConfig``, ``MatchConfig``) or the raw PyO3 classes interchangeably.
The raw PyO3 types remain accessible via ``corrmatch._corrmatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from . import _corrmatch as _c

if TYPE_CHECKING:
    from .config import CompileConfig, MatchConfig


def _to_rust_compile(
    cfg: Union["CompileConfig", _c.CompileConfig, None],
) -> _c.CompileConfig | None:
    """Coerce a Python dataclass or PyO3 CompileConfig to the PyO3 type."""
    if cfg is None or isinstance(cfg, _c.CompileConfig):
        return cfg
    # cfg is narrowed to CompileConfig by the isinstance guard above
    return cfg.to_rust()


def _to_rust_match(
    cfg: Union["MatchConfig", _c.MatchConfig, None],
) -> _c.MatchConfig | None:
    """Coerce a Python dataclass or PyO3 MatchConfig to the PyO3 type."""
    if cfg is None or isinstance(cfg, _c.MatchConfig):
        return cfg
    # cfg is narrowed to MatchConfig by the isinstance guard above
    return cfg.to_rust()


@dataclass(frozen=True, slots=True)
class Match:
    """Frozen match result with position, angle, and score.

    Attributes:
        x: Top-left x coordinate of the template placement (subpixel).
        y: Top-left y coordinate of the template placement (subpixel).
        angle_deg: Estimated rotation angle in degrees.
        score: Match score (ZNCC in [-1, 1], or negative SSE for SSD).
    """

    x: float
    y: float
    angle_deg: float
    score: float

    @classmethod
    def from_rust(cls, m: _c.Match) -> "Match":
        """Construct a ``Match`` from a raw PyO3 ``Match`` object."""
        return cls(x=m.x, y=m.y, angle_deg=m.angle_deg, score=m.score)


class Matcher:
    """Thin Python wrapper around the PyO3 ``Matcher``.

    Runs coarse-to-fine template matching and returns Python-native ``Match``
    objects.

    Created via ``CompiledTemplate.matcher()``.
    """

    def __init__(self, inner: _c.Matcher) -> None:
        self._inner = inner

    def match_image(self, image: NDArray[np.uint8]) -> Match:
        """Find the best match in an image.

        Args:
            image: 2D uint8 numpy array (height × width).

        Returns:
            Best ``Match`` result.

        Raises:
            RuntimeError: If matching fails.
        """
        return Match.from_rust(self._inner.match_image(image))

    def match_topk(self, image: NDArray[np.uint8], k: int) -> list[Match]:
        """Find the top-K matches in an image.

        Args:
            image: 2D uint8 numpy array (height × width).
            k: Number of matches to return.

        Returns:
            List of up to ``k`` ``Match`` results, sorted by score descending.

        Raises:
            RuntimeError: If matching fails.
        """
        return [Match.from_rust(m) for m in self._inner.match_topk(image, k)]

    def __repr__(self) -> str:
        return repr(self._inner)


class CompiledTemplate:
    """Thin Python wrapper around the PyO3 ``CompiledTemplate``.

    Created via ``Template.compile()`` or ``Template.compile_no_rotation()``.
    """

    def __init__(self, inner: _c.CompiledTemplate) -> None:
        self._inner = inner

    @property
    def num_levels(self) -> int:
        """Number of pyramid levels in the compiled template."""
        return self._inner.num_levels

    def matcher(
        self,
        cfg: Union["MatchConfig", _c.MatchConfig, None] = None,
    ) -> Matcher:
        """Create a ``Matcher`` with the given configuration.

        Accepts either a Python-native :class:`corrmatch.config.MatchConfig`
        dataclass or a raw PyO3 ``_corrmatch.MatchConfig``.

        Args:
            cfg: Match configuration (default: ``MatchConfig()``).

        Returns:
            A ready-to-use ``Matcher``.
        """
        rust_cfg = _to_rust_match(cfg)
        return Matcher(self._inner.matcher(rust_cfg))

    def __repr__(self) -> str:
        return repr(self._inner)


class Template:
    """Thin Python wrapper around the PyO3 ``Template``.

    Args:
        data: 2D uint8 numpy array (height × width).
    """

    def __init__(self, data: NDArray[np.uint8]) -> None:
        self._inner = _c.Template(data)

    @property
    def width(self) -> int:
        """Width of the template in pixels."""
        return self._inner.width

    @property
    def height(self) -> int:
        """Height of the template in pixels."""
        return self._inner.height

    def compile(
        self,
        cfg: Union["CompileConfig", _c.CompileConfig, None] = None,
    ) -> CompiledTemplate:
        """Compile with rotation support.

        Accepts either a Python-native :class:`corrmatch.config.CompileConfig`
        dataclass or a raw PyO3 ``_corrmatch.CompileConfig``.

        Args:
            cfg: Compile configuration (default: ``CompileConfig()``).

        Returns:
            Compiled template assets ready for matching.
        """
        rust_cfg = _to_rust_compile(cfg)
        return CompiledTemplate(self._inner.compile(rust_cfg))

    def compile_no_rotation(self, max_levels: int = 6) -> CompiledTemplate:
        """Compile without rotation support (faster).

        Args:
            max_levels: Maximum pyramid levels (default: 6).

        Returns:
            Compiled template assets for translation-only matching.
        """
        return CompiledTemplate(self._inner.compile_no_rotation(max_levels))

    def __repr__(self) -> str:
        return repr(self._inner)
