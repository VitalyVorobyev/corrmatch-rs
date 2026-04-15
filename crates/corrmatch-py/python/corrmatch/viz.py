"""Visualization helpers for CorrMatch results.

This module provides an interactive matplotlib figure for inspecting template
matching results. It is intended for human-in-the-loop debugging and does not
affect matching behavior.

Requires:
    - numpy
    - matplotlib (recommended: install via `pip install corrmatch[viz]`)
    - Pillow (only needed for the CLI image loader)
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import corrmatch


@dataclass(frozen=True)
class MatchRecord:
    """Lightweight match record (useful for JSON inputs)."""

    x: float
    y: float
    angle_deg: float
    score: float


def _require_2d_u8(name: str, arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got dtype={arr.dtype}")
    return arr


def _match_fields(m: Any) -> MatchRecord:
    if isinstance(m, MatchRecord):
        return m
    return MatchRecord(
        x=float(getattr(m, "x")),
        y=float(getattr(m, "y")),
        angle_deg=float(getattr(m, "angle_deg")),
        score=float(getattr(m, "score")),
    )


def _sorted_matches(matches: Sequence[Any], max_show: int) -> list[MatchRecord]:
    if max_show <= 0:
        return []
    out = [_match_fields(m) for m in matches]
    out.sort(key=lambda m: m.score, reverse=True)
    return out[:max_show]


def _rot_cw(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 2D points by a clockwise angle in image coordinates."""
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # In image coordinates (x right, y down), a positive clockwise rotation has
    # the same matrix form as a standard CCW rotation in y-up coordinates.
    r = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    result: np.ndarray = points @ r.T
    return result


def rotated_rect_corners(
    x: float, y: float, width: int, height: int, angle_deg: float
) -> np.ndarray:
    """Compute rotated rectangle corners for a template placement.

    The returned corners are in image coordinates (x right, y down), matching
    corrmatch's convention: positive `angle_deg` rotates clockwise.
    """
    w = float(width)
    h = float(height)

    # Use pixel-edge-aligned corners for a crisp box.
    corners = np.array(
        [
            [-0.5, -0.5],
            [w - 0.5, -0.5],
            [w - 0.5, h - 0.5],
            [-0.5, h - 0.5],
        ],
        dtype=np.float32,
    )
    center = np.array([(w - 1.0) * 0.5, (h - 1.0) * 0.5], dtype=np.float32)
    rotated = _rot_cw(corners - center, angle_deg) + center
    rotated[:, 0] += float(x)
    rotated[:, 1] += float(y)
    return rotated


def _extract_patch_u8(
    image: np.ndarray, x0: int, y0: int, width: int, height: int, fill_value: int
) -> tuple[np.ndarray, np.ndarray]:
    img_h, img_w = image.shape
    patch = np.full((height, width), int(fill_value), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(img_w, x0 + width)
    src_y1 = min(img_h, y0 + height)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return patch, mask

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    patch[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    mask[dst_y0:dst_y1, dst_x0:dst_x1] = 1
    return patch, mask


def show_matches(
    image: np.ndarray,
    template: np.ndarray,
    matches: Sequence[Any],
    *,
    match_cfg: corrmatch.MatchConfig | None = None,
    compile_cfg: corrmatch.CompileConfig | None = None,
    max_show: int = 5,
    fill_value: int = 0,
    title: str | None = None,
    show: bool = True,
) -> Any:
    """Show an interactive matplotlib figure for the given matches.

    Args:
        image: 2D uint8 image array (H, W)
        template: 2D uint8 template array (h, w)
        matches: Sequence of match objects (corrmatch.Match or MatchRecord-like)
        match_cfg: Optional MatchConfig used (for display only)
        compile_cfg: Optional CompileConfig used (for display only)
        max_show: Number of top matches to overlay on the image
        fill_value: Fill value used for rotations/padding in visualization
        title: Optional figure title
        show: If True, calls matplotlib `show()`

    Returns:
        The created matplotlib Figure.
    """
    _require_2d_u8("image", image)
    _require_2d_u8("template", template)

    top = _sorted_matches(matches, max_show=max_show)
    if not top:
        raise ValueError("matches is empty")
    best = top[0]

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as e:  # pragma: no cover - exercised only when matplotlib missing
        raise ImportError(
            "corrmatch.viz requires matplotlib; install with `pip install corrmatch[viz]`"
        ) from e

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        gridspec_kw={"width_ratios": [2.2, 1.0, 1.0]},
        constrained_layout=True,
    )
    ax_img, ax_tpl, ax_tpl_rot = axes[0]
    ax_patch, ax_deskew, ax_diff = axes[1]

    # Image + overlays
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax_img.set_title("Image (top-K overlays)")

    tpl_h, tpl_w = template.shape
    colors = ["lime", "cyan", "yellow", "magenta", "orange"]
    for i, m in enumerate(top):
        corners = rotated_rect_corners(m.x, m.y, tpl_w, tpl_h, m.angle_deg)
        is_best = i == 0
        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=colors[min(i, len(colors) - 1)],
            linewidth=2.0 if is_best else 1.2,
            linestyle="-" if is_best else "--",
        )
        ax_img.add_patch(poly)

        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))
        ax_img.text(
            cx,
            cy,
            f"{i+1}",
            color=poly.get_edgecolor(),
            fontsize=10,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
        )

    info_lines = [
        f"best: x={best.x:.2f}, y={best.y:.2f}",
        f"angle={best.angle_deg:.2f}° (cw), score={best.score:.4f}",
        f"template: {tpl_w}×{tpl_h}",
    ]
    if match_cfg is not None:
        info_lines.append(
            f"match: metric={match_cfg.metric}, rotation={match_cfg.rotation}, "
            f"beam={match_cfg.beam_width}, nms={match_cfg.nms_radius}, roi={match_cfg.roi_radius}"
        )
    if compile_cfg is not None:
        info_lines.append(
            f"compile: levels={compile_cfg.max_levels}, coarse_step={compile_cfg.coarse_step_deg}°"
        )

    ax_img.text(
        0.01,
        0.01,
        "\n".join(info_lines),
        transform=ax_img.transAxes,
        color="white",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 4, "edgecolor": "none"},
    )

    # Template
    ax_tpl.imshow(template, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax_tpl.set_title("Template")

    # Rotated template (best angle)
    rot_tpl, rot_mask = corrmatch.rotate_u8_bilinear_masked(
        template, angle_deg=best.angle_deg, fill_value=int(fill_value)
    )
    ax_tpl_rot.imshow(rot_tpl, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax_tpl_rot.imshow(
        np.where(rot_mask != 0, 1.0, np.nan),
        cmap="Reds",
        alpha=0.20,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    ax_tpl_rot.set_title("Template rotated (best)")

    # Patch at best placement (rounded to nearest pixel for display)
    x0 = int(math.floor(best.x + 0.5))
    y0 = int(math.floor(best.y + 0.5))
    patch, patch_mask = _extract_patch_u8(
        image, x0=x0, y0=y0, width=tpl_w, height=tpl_h, fill_value=int(fill_value)
    )
    ax_patch.imshow(patch, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax_patch.set_title("Matched patch (axis-aligned)")

    # Deskew patch back to 0 deg (rotate counter-clockwise => negative cw angle)
    deskew, deskew_mask = corrmatch.rotate_u8_bilinear_masked(
        patch, angle_deg=-best.angle_deg, fill_value=int(fill_value)
    )
    ax_deskew.imshow(deskew, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax_deskew.imshow(
        np.where(deskew_mask != 0, 1.0, np.nan),
        cmap="Reds",
        alpha=0.20,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    ax_deskew.set_title("Matched patch (deskewed)")

    # Diff: deskewed patch vs template (masked)
    valid = (deskew_mask != 0) & (patch_mask != 0)
    diff = np.abs(deskew.astype(np.int16) - template.astype(np.int16)).astype(np.float32)
    diff_ma = np.ma.array(diff, mask=~valid)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0.0)
    ax_diff.imshow(diff_ma, cmap=cmap, origin="upper", vmin=0.0, vmax=255.0)
    ax_diff.set_title("|deskew - template|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)

    # Expose axes handles for external tools (e.g. saving a single panel).
    fig.corrmatch_axes = {
        "image": ax_img,
        "template": ax_tpl,
        "template_rotated": ax_tpl_rot,
        "patch": ax_patch,
        "deskew": ax_deskew,
        "diff": ax_diff,
    }

    if show:
        plt.show()

    return fig


def match_and_show(
    image: np.ndarray,
    template: np.ndarray,
    *,
    topk: int = 5,
    match_cfg: corrmatch.MatchConfig | None = None,
    compile_cfg: corrmatch.CompileConfig | None = None,
    fill_value: int = 0,
    title: str | None = None,
    show: bool = True,
) -> tuple[Any, list[corrmatch.Match]]:
    """Run CorrMatch and immediately visualize the top-K results."""
    _require_2d_u8("image", image)
    _require_2d_u8("template", template)
    if topk <= 0:
        raise ValueError("topk must be >= 1")

    rotation = match_cfg.rotation if match_cfg is not None else "disabled"

    tpl = corrmatch.Template(template)
    if rotation == "enabled":
        compiled = tpl.compile(compile_cfg or corrmatch.CompileConfig())
    else:
        max_levels = (compile_cfg.max_levels if compile_cfg is not None else 6)
        compiled = tpl.compile_no_rotation(max_levels=max_levels)

    matcher = compiled.matcher(match_cfg)
    matches = matcher.match_topk(image, k=topk)

    fig = show_matches(
        image,
        template,
        matches,
        match_cfg=match_cfg,
        compile_cfg=compile_cfg,
        max_show=topk,
        fill_value=fill_value,
        title=title,
        show=show,
    )
    return fig, matches


def load_matches_json(path: str | Path) -> list[MatchRecord]:
    """Load matches from corrmatch-cli JSON output."""
    p = Path(path)
    payload = json.loads(p.read_text())
    topk = payload.get("topk", [])
    if isinstance(topk, int):
        raise ValueError(
            "Expected corrmatch-cli output JSON with 'topk' as an array, but got a config-like JSON "
            "(where 'topk' is an integer)."
        )
    if not isinstance(topk, list):
        raise ValueError(
            f"Expected corrmatch-cli output JSON with 'topk' as a list, got type={type(topk).__name__}"
        )
    out: list[MatchRecord] = []
    for item in topk:
        out.append(
            MatchRecord(
                x=float(item["x"]),
                y=float(item["y"]),
                angle_deg=float(item["angle_deg"]),
                score=float(item["score"]),
            )
        )
    out.sort(key=lambda m: m.score, reverse=True)
    return out


def show_from_json(
    image: np.ndarray,
    template: np.ndarray,
    match_json_path: str | Path,
    *,
    max_show: int = 5,
    fill_value: int = 0,
    title: str | None = None,
    show: bool = True,
) -> Any:
    """Visualize matches produced elsewhere (e.g. corrmatch-cli)."""
    matches = load_matches_json(match_json_path)
    return show_matches(
        image,
        template,
        matches,
        max_show=max_show,
        fill_value=fill_value,
        title=title,
        show=show,
    )


def _load_gray_u8(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - exercised only when Pillow missing
        raise ImportError(
            "corrmatch.viz CLI requires Pillow; install with `pip install corrmatch[viz]`"
        ) from e

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got shape={arr.shape}")
    return arr


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corrmatch.viz")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--template", required=True, help="Path to template image")
    parser.add_argument("--match-json", help="Path to corrmatch-cli JSON output (optional)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K overlays")
    parser.add_argument(
        "--rotation",
        choices=["enabled", "disabled"],
        default="disabled",
        help="Enable rotation search (when running matching)",
    )
    parser.add_argument(
        "--metric",
        choices=["zncc", "ssd"],
        default="zncc",
        help="Matching metric (when running matching)",
    )
    parser.add_argument(
        "--fill-value",
        type=int,
        default=0,
        help="Fill value used for visualization rotations",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    image = _load_gray_u8(args.image)
    template = _load_gray_u8(args.template)

    if args.match_json:
        show_from_json(
            image,
            template,
            args.match_json,
            max_show=args.topk,
            fill_value=args.fill_value,
            title=f"corrmatch (from JSON): {Path(args.match_json).name}",
            show=True,
        )
        return 0

    match_cfg = corrmatch.MatchConfig(metric=args.metric, rotation=args.rotation)
    match_and_show(
        image,
        template,
        topk=args.topk,
        match_cfg=match_cfg,
        compile_cfg=None,
        fill_value=args.fill_value,
        title=f"corrmatch: rotation={args.rotation}, metric={args.metric}",
        show=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
