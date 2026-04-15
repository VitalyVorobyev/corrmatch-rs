"""Python-native configuration dataclasses for corrmatch.

These dataclasses provide a typed, IDE-friendly interface on top of the
raw PyO3-generated classes in ``corrmatch._corrmatch``.  They are the
recommended way to configure corrmatch from Python.

Example:
    >>> from corrmatch.config import CompileConfig, MatchConfig
    >>> compile_cfg = CompileConfig(max_levels=4, coarse_step_deg=15.0)
    >>> match_cfg = MatchConfig(rotation="enabled", beam_width=8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import _corrmatch as _c

MetricLiteral = Literal["zncc", "ssd"]
RotationLiteral = Literal["enabled", "disabled"]


@dataclass(frozen=True, kw_only=True)
class CompileConfig:
    """Configuration for compiling a template with rotation support.

    This dataclass wraps ``corrmatch._corrmatch.CompileConfig`` and adds
    Python-style keyword-only construction with typed fields.

    Args:
        max_levels: Maximum pyramid levels to build. Must be >= 1.
            Defaults to 6. Fewer levels speed up compilation and matching
            at the cost of coarser candidates at the coarsest level.
        coarse_step_deg: Coarse rotation step in degrees at the finest
            angle-grid level. Must be a positive finite value.
            Defaults to 10.0 degrees.
        min_step_deg: Minimum rotation step in degrees across pyramid
            levels. Must satisfy ``0 < min_step_deg <= coarse_step_deg``.
            Defaults to 0.5 degrees.
        fill_value: Fill value (0–255) used for out-of-bounds rotations.
            Defaults to 0.
        precompute_coarsest: When ``True``, all angle slots at the
            coarsest pyramid level are precomputed at compile time.
            Defaults to ``True``.

    Example:
        >>> from corrmatch.config import CompileConfig
        >>> cfg = CompileConfig(max_levels=3, coarse_step_deg=15.0, min_step_deg=5.0)
        >>> rust_cfg = cfg.to_rust()

    Raises:
        ValueError: If any parameter fails validation (delegated to the
            Rust ``CompileConfig.validate()`` method).
    """

    max_levels: int = 6
    coarse_step_deg: float = 10.0
    min_step_deg: float = 0.5
    fill_value: int = 0
    precompute_coarsest: bool = True

    def __post_init__(self) -> None:
        """Validate fields by delegating to the Rust-side validator."""
        self.to_rust()

    def to_rust(self) -> _c.CompileConfig:
        """Convert to a ``corrmatch._corrmatch.CompileConfig`` instance.

        Returns:
            The raw Rust-backed config object, ready for use with
            ``Template.compile()``.

        Raises:
            ValueError: If validation fails.
        """
        return _c.CompileConfig(
            max_levels=self.max_levels,
            coarse_step_deg=self.coarse_step_deg,
            min_step_deg=self.min_step_deg,
            fill_value=self.fill_value,
            precompute_coarsest=self.precompute_coarsest,
        )

    @classmethod
    def from_rust(cls, cfg: _c.CompileConfig) -> "CompileConfig":
        """Construct from a ``corrmatch._corrmatch.CompileConfig`` instance.

        Args:
            cfg: The raw Rust-backed config object.

        Returns:
            A new ``CompileConfig`` dataclass instance.
        """
        return cls(
            max_levels=cfg.max_levels,
            coarse_step_deg=cfg.coarse_step_deg,
            min_step_deg=cfg.min_step_deg,
            fill_value=cfg.fill_value,
            precompute_coarsest=cfg.precompute_coarsest,
        )


@dataclass(frozen=True, kw_only=True)
class MatchConfig:
    """Configuration for the coarse-to-fine matcher pipeline.

    This dataclass wraps ``corrmatch._corrmatch.MatchConfig`` and adds
    Python-style keyword-only construction with typed ``Literal`` fields.

    Args:
        metric: Matching metric. ``"zncc"`` (Zero-Mean Normalized Cross-
            Correlation, higher is better, roughly in [-1, 1]) or ``"ssd"``
            (Sum of Squared Differences reported as negative SSE, higher is
            better). Defaults to ``"zncc"``.
        rotation: Whether rotation search is enabled. ``"enabled"``
            requires a template compiled with rotation support; ``"disabled"``
            uses the faster unmasked translation-only path. Defaults to
            ``"disabled"``.
        parallel: Enable parallel search using Rayon thread pool (requires
            the ``rayon`` Cargo feature). Defaults to ``False``.
        max_image_levels: Maximum image pyramid levels. Defaults to 6.
        beam_width: Number of candidates kept per level after merge and
            NMS. Defaults to 8.
        per_angle_topk: Top-M peaks per angle at the coarsest level
            (rotation-enabled path only). Defaults to 3.
        nms_radius: Spatial NMS radius in pixels. Defaults to 6.
        roi_radius: Refinement ROI radius in pixels. Defaults to 8.
        angle_half_range_steps: Angle neighbourhood half-range in multiples
            of the grid step (rotation path only). Defaults to 1.
        min_var_i: Minimum image patch variance; patches below this
            threshold are skipped (ZNCC only). Defaults to 1e-8.
        min_score: Minimum score threshold; candidates below this value are
            discarded. Defaults to negative infinity (keep all).

    Example:
        >>> from corrmatch.config import MatchConfig
        >>> cfg = MatchConfig(rotation="enabled", beam_width=12, min_score=0.7)
        >>> rust_cfg = cfg.to_rust()

    Raises:
        ValueError: If any parameter fails Rust-side validation.
    """

    metric: MetricLiteral = "zncc"
    rotation: RotationLiteral = "disabled"
    parallel: bool = False
    max_image_levels: int = 6
    beam_width: int = 8
    per_angle_topk: int = 3
    nms_radius: int = 6
    roi_radius: int = 8
    angle_half_range_steps: int = 1
    min_var_i: float = 1e-8
    min_score: float = float("-inf")

    def __post_init__(self) -> None:
        """Validate fields by delegating to the Rust-side validator."""
        self.to_rust()

    def to_rust(self) -> _c.MatchConfig:
        """Convert to a ``corrmatch._corrmatch.MatchConfig`` instance.

        Returns:
            The raw Rust-backed config object, ready for use with
            ``CompiledTemplate.matcher()``.

        Raises:
            ValueError: If validation fails.
        """
        return _c.MatchConfig(
            metric=self.metric,
            rotation=self.rotation,
            parallel=self.parallel,
            max_image_levels=self.max_image_levels,
            beam_width=self.beam_width,
            per_angle_topk=self.per_angle_topk,
            nms_radius=self.nms_radius,
            roi_radius=self.roi_radius,
            angle_half_range_steps=self.angle_half_range_steps,
            min_var_i=self.min_var_i,
            min_score=self.min_score,
        )

    @classmethod
    def from_rust(cls, cfg: _c.MatchConfig) -> "MatchConfig":
        """Construct from a ``corrmatch._corrmatch.MatchConfig`` instance.

        Args:
            cfg: The raw Rust-backed config object.

        Returns:
            A new ``MatchConfig`` dataclass instance.
        """
        return cls(
            metric=cfg.metric,  # type: ignore[arg-type]
            rotation=cfg.rotation,  # type: ignore[arg-type]
            parallel=cfg.parallel,
            max_image_levels=cfg.max_image_levels,
            beam_width=cfg.beam_width,
            per_angle_topk=cfg.per_angle_topk,
            nms_radius=cfg.nms_radius,
            roi_radius=cfg.roi_radius,
            angle_half_range_steps=cfg.angle_half_range_steps,
            min_var_i=cfg.min_var_i,
            min_score=cfg.min_score,
        )
