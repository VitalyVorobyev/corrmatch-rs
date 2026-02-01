#!/usr/bin/env python3
"""Run CorrMatch and open an interactive visualization window.

This is a convenience wrapper around the `corrmatch` maturin bindings and
`corrmatch.viz` helper module.

Examples:
  - From paths:
      python tools/viz_detect.py --image image.png --template tpl.png --rotation enabled

  - Using a corrmatch-cli JSON config (same fields):
      python tools/viz_detect.py --config corrmatch-cli/config.example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_gray_u8(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit(
            "Pillow is required for tools/viz_detect.py. Install via:\n"
            "  pip install -e corrmatch-py[viz]\n"
            "or:\n"
            "  pip install Pillow\n"
        ) from e

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 2:
        raise SystemExit(f"expected 2D grayscale image, got shape={arr.shape}")
    return arr


def _get(dct: dict[str, Any], key: str, default: Any) -> Any:
    val = dct.get(key, default)
    return default if val is None else val


def _save_axes_png(fig: Any, ax: Any, path: str | Path, *, dpi: int = 150) -> None:
    # Matplotlib computes extents after the first draw.
    fig.canvas.draw()

    title = ax.get_title()
    ax.set_title("")
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi=dpi, bbox_inches=bbox, pad_inches=0.0)

    ax.set_title(title)


def main() -> int:
    parser = argparse.ArgumentParser(prog="viz_detect.py")
    parser.add_argument("--config", help="Path to corrmatch-cli style JSON config")
    parser.add_argument("--image", help="Path to input image (overrides config)")
    parser.add_argument("--template", help="Path to template image (overrides config)")
    parser.add_argument(
        "--match-json",
        help=(
            "Path to corrmatch-cli JSON output (best/topk) OR a corrmatch-cli style config JSON "
            "(compile/match/topk)."
        ),
    )
    parser.add_argument("--topk", type=int, help="Top-K matches to show")

    parser.add_argument(
        "--rotation",
        choices=["enabled", "disabled"],
        help="Rotation mode (when running matching)",
    )
    parser.add_argument("--metric", choices=["zncc", "ssd"], help="Metric (when running matching)")
    parser.add_argument("--parallel", action="store_true", help="Enable rayon parallel matching")
    parser.add_argument("--fill-value", type=int, default=0, help="Fill value for visualization")
    parser.add_argument("--save", help="Optional output path to save the figure (png/svg/pdf)")
    parser.add_argument(
        "--save-overlay",
        help="Optional output path to save ONLY the top-left overlay axes as PNG (no title).",
    )
    args = parser.parse_args()

    try:
        import corrmatch
        import corrmatch.viz as viz
    except Exception as e:
        raise SystemExit(
            "Failed to import corrmatch bindings.\n"
            "Build/install them first, e.g.:\n"
            "  cd corrmatch-py && maturin develop --release -i $(python3 -c 'import sys; print(sys.executable)')\n"
        ) from e

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = json.loads(Path(args.config).read_text())

    image_path = args.image or cfg.get("image_path")
    template_path = args.template or cfg.get("template_path")
    if not image_path or not template_path:
        raise SystemExit("Provide --image and --template (or --config with image_path/template_path).")

    image = _load_gray_u8(image_path)
    template = _load_gray_u8(template_path)

    if args.match_json:
        payload = json.loads(Path(args.match_json).read_text())
        topk_payload = payload.get("topk")
        is_output_json = (
            isinstance(topk_payload, list)
            and (len(topk_payload) == 0 or (isinstance(topk_payload[0], dict) and "x" in topk_payload[0]))
        ) or ("best" in payload and isinstance(payload.get("best"), dict))

        if is_output_json:
            fig = viz.show_from_json(
                image,
                template,
                args.match_json,
                max_show=args.topk or _get(cfg, "topk", 5),
                fill_value=args.fill_value,
                title=f"corrmatch (from output JSON): {Path(args.match_json).name}",
                show=False,
            )
            ax_img = getattr(fig, "corrmatch_axes", {}).get("image") or fig.axes[0]
            if args.save_overlay:
                _save_axes_png(fig, ax_img, args.save_overlay, dpi=150)
            if args.save:
                fig.savefig(args.save, dpi=150)
            import matplotlib.pyplot as plt

            plt.show()
            return 0

        # Treat as config-like JSON: use it as the configuration source and run matching.
        cfg = payload

    compile_cfg_raw = cfg.get("compile", {})
    match_cfg_raw = cfg.get("match", cfg.get("match_cfg", {}))

    rotation = args.rotation or _get(match_cfg_raw, "rotation", "disabled")
    metric = args.metric or _get(match_cfg_raw, "metric", "zncc")
    topk = args.topk or int(_get(cfg, "topk", 5))

    compile_cfg = corrmatch.CompileConfig(
        max_levels=int(_get(compile_cfg_raw, "max_levels", 6)),
        coarse_step_deg=float(_get(compile_cfg_raw, "coarse_step_deg", 10.0)),
        min_step_deg=float(_get(compile_cfg_raw, "min_step_deg", 0.5)),
        fill_value=int(_get(compile_cfg_raw, "fill_value", 0)),
        precompute_coarsest=bool(_get(compile_cfg_raw, "precompute_coarsest", True)),
    )

    match_cfg = corrmatch.MatchConfig(
        metric=str(metric),
        rotation=str(rotation),
        parallel=bool(args.parallel or _get(match_cfg_raw, "parallel", False)),
        max_image_levels=int(_get(match_cfg_raw, "max_image_levels", 6)),
        beam_width=int(_get(match_cfg_raw, "beam_width", 8)),
        per_angle_topk=int(_get(match_cfg_raw, "per_angle_topk", 3)),
        nms_radius=int(_get(match_cfg_raw, "nms_radius", 6)),
        roi_radius=int(_get(match_cfg_raw, "roi_radius", 8)),
        angle_half_range_steps=int(_get(match_cfg_raw, "angle_half_range_steps", 1)),
        min_var_i=float(_get(match_cfg_raw, "min_var_i", 1e-8)),
        min_score=float(_get(match_cfg_raw, "min_score", float("-inf"))),
    )

    fig, _matches = viz.match_and_show(
        image,
        template,
        topk=topk,
        match_cfg=match_cfg,
        compile_cfg=compile_cfg,
        fill_value=args.fill_value,
        title=f"corrmatch: rotation={rotation}, metric={metric}, topk={topk}",
        show=False,
    )
    ax_img = getattr(fig, "corrmatch_axes", {}).get("image") or fig.axes[0]
    if args.save_overlay:
        _save_axes_png(fig, ax_img, args.save_overlay, dpi=150)
    if args.save:
        fig.savefig(args.save, dpi=150)
    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
