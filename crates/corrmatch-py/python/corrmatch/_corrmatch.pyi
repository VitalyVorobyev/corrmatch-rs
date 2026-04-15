"""Type stubs for the corrmatch._corrmatch extension module.

This module is the raw PyO3-generated extension. Prefer the Python-native
wrappers in ``corrmatch`` (``CompileConfig``, ``MatchConfig``) for new code.
"""

from typing import Tuple
from numpy.typing import NDArray
import numpy as np

__version__: str

class Match:
    """Match result containing position, angle, and score.

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
    def __repr__(self) -> str: ...

class CompileConfig:
    """Raw Rust-backed compile configuration with rotation support.

    Args:
        max_levels: Maximum pyramid levels (default: 6).
        coarse_step_deg: Coarse rotation step in degrees (default: 10.0).
        min_step_deg: Minimum rotation step in degrees (default: 0.5).
        fill_value: Out-of-bounds fill value (default: 0).
        precompute_coarsest: Precompute coarsest level (default: True).
    """

    def __init__(
        self,
        max_levels: int = 6,
        coarse_step_deg: float = 10.0,
        min_step_deg: float = 0.5,
        fill_value: int = 0,
        precompute_coarsest: bool = True,
    ) -> None: ...
    @property
    def max_levels(self) -> int: ...
    @property
    def coarse_step_deg(self) -> float: ...
    @property
    def min_step_deg(self) -> float: ...
    @property
    def fill_value(self) -> int: ...
    @property
    def precompute_coarsest(self) -> bool: ...
    def validate(self) -> None: ...
    def __repr__(self) -> str: ...

class MatchConfig:
    """Raw Rust-backed match configuration.

    Args:
        metric: ``"zncc"`` or ``"ssd"`` (default: ``"zncc"``).
        rotation: ``"enabled"`` or ``"disabled"`` (default: ``"disabled"``).
        parallel: Enable parallel execution (default: False).
        max_image_levels: Maximum image pyramid levels (default: 6).
        beam_width: Candidates kept per level (default: 8).
        per_angle_topk: Top-K peaks per angle at coarsest level (default: 3).
        nms_radius: Spatial NMS radius in pixels (default: 6).
        roi_radius: Refinement ROI radius in pixels (default: 8).
        angle_half_range_steps: Angle neighbourhood half-range (default: 1).
        min_var_i: Minimum image patch variance (default: 1e-8).
        min_score: Minimum score threshold (default: -inf).
    """

    def __init__(
        self,
        metric: str = "zncc",
        rotation: str = "disabled",
        parallel: bool = False,
        max_image_levels: int = 6,
        beam_width: int = 8,
        per_angle_topk: int = 3,
        nms_radius: int = 6,
        roi_radius: int = 8,
        angle_half_range_steps: int = 1,
        min_var_i: float = 1e-8,
        min_score: float = ...,
    ) -> None: ...
    @property
    def metric(self) -> str: ...
    @property
    def rotation(self) -> str: ...
    @property
    def parallel(self) -> bool: ...
    @property
    def max_image_levels(self) -> int: ...
    @property
    def beam_width(self) -> int: ...
    @property
    def per_angle_topk(self) -> int: ...
    @property
    def nms_radius(self) -> int: ...
    @property
    def roi_radius(self) -> int: ...
    @property
    def angle_half_range_steps(self) -> int: ...
    @property
    def min_var_i(self) -> float: ...
    @property
    def min_score(self) -> float: ...
    def validate(self) -> None: ...
    def __repr__(self) -> str: ...

class Template:
    """Owned template image.

    Args:
        data: 2D uint8 numpy array (height × width).
    """

    def __init__(self, data: NDArray[np.uint8]) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def compile(self, cfg: CompileConfig | None = None) -> CompiledTemplate:
        """Compile with rotation support.

        Args:
            cfg: Compile configuration (default: ``CompileConfig()``).

        Returns:
            Compiled template assets ready for matching.
        """
        ...
    def compile_no_rotation(self, max_levels: int = 6) -> CompiledTemplate:
        """Compile without rotation support (faster).

        Args:
            max_levels: Maximum pyramid levels (default: 6).

        Returns:
            Compiled template assets for translation-only matching.
        """
        ...
    def __repr__(self) -> str: ...

class CompiledTemplate:
    """Compiled template assets (opaque).

    Created via ``Template.compile()`` or ``Template.compile_no_rotation()``.
    """

    @property
    def num_levels(self) -> int: ...
    def matcher(self, cfg: MatchConfig | None = None) -> Matcher:
        """Create a matcher with the given configuration.

        Args:
            cfg: Match configuration (default: ``MatchConfig()``).

        Returns:
            A ready-to-use ``Matcher``.
        """
        ...
    def __repr__(self) -> str: ...

class Matcher:
    """Runs coarse-to-fine template matching.

    Created via ``CompiledTemplate.matcher()``.
    """

    def match_image(self, image: NDArray[np.uint8]) -> Match:
        """Find the best match in an image.

        Args:
            image: 2D uint8 numpy array (height × width).

        Returns:
            Best ``Match`` result.

        Raises:
            RuntimeError: If matching fails.
        """
        ...
    def match_topk(
        self, image: NDArray[np.uint8], k: int
    ) -> list[Match]:
        """Find the top-K matches in an image.

        Args:
            image: 2D uint8 numpy array (height × width).
            k: Number of matches to return.

        Returns:
            List of up to ``k`` ``Match`` results, sorted by score descending.

        Raises:
            RuntimeError: If matching fails.
        """
        ...
    def __repr__(self) -> str: ...

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
    ...

def rotate_u8_bilinear_masked(
    image: NDArray[np.uint8],
    angle_deg: float,
    fill_value: int = 0,
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Rotate a grayscale image using bilinear sampling with a validity mask.

    Args:
        image: 2D uint8 numpy array (height × width).
        angle_deg: Rotation angle in degrees (clockwise in image coordinates).
        fill_value: Fill value for out-of-bounds pixels (default: 0).

    Returns:
        Tuple ``(rotated, mask)`` where ``rotated`` is the rotated image
        (same shape as ``image``) and ``mask`` is a binary uint8 array
        with 1 at pixels where bilinear interpolation is fully inside
        bounds and 0 elsewhere.
    """
    ...
