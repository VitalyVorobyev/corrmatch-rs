#!/usr/bin/env python3
import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover - runtime guard
    print(
        "Pillow is required. Activate the venv and install:\n"
        "  pip install -r tools/synth_cases/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)


DEFAULT_COMPILE = {
    "max_levels": 4,
    "coarse_step_deg": 30.0,
    "min_step_deg": 7.5,
    "fill_value": 0,
    "precompute_coarsest": True,
}

DEFAULT_MATCH = {
    "metric": "zncc",
    "rotation": "disabled",
    "parallel": False,
    "max_image_levels": 4,
    "beam_width": 6,
    "per_angle_topk": 3,
    "nms_radius": 4,
    "roi_radius": 6,
    "angle_half_range_steps": 1,
    "min_var_i": 1e-8,
    "min_score": -1.0e38,
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: str
    image_size: Tuple[int, int]
    template_size: Tuple[int, int]
    template_pattern: str
    background_style: str
    rotation_deg: float
    present: bool = True
    template_gain: float = 1.0
    template_bias: float = 0.0
    global_gain: float = 1.0
    global_bias: float = 0.0
    noise_sigma: float = 0.0
    blur_sigma: float = 0.0
    salt_pepper: float = 0.0
    occlusion_frac: float = 0.0
    distractors: int = 0
    distractor_jitter_deg: float = 0.0
    place_mode: str = "random"
    background_value: Optional[int] = None
    compile_overrides: Optional[Dict[str, object]] = None
    match_overrides: Optional[Dict[str, object]] = None
    topk: Optional[int] = None
    notes: str = ""


def clamp_u8(value: float) -> int:
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return int(value)


def sin_cos_deg(angle_deg: float) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    return math.sin(rad), math.cos(rad)


def rotate_u8_bilinear_masked(
    src: List[int], width: int, height: int, angle_deg: float, fill: int
) -> Tuple[List[int], List[int]]:
    out = [fill] * (width * height)
    mask = [0] * (width * height)
    if width < 2 or height < 2:
        return out, mask

    sin_a, cos_a = sin_cos_deg(angle_deg)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    max_x = float(width - 1)
    max_y = float(height - 1)

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            src_x = cos_a * dx + sin_a * dy + cx
            src_y = -sin_a * dx + cos_a * dy + cy

            if not math.isfinite(src_x) or not math.isfinite(src_y):
                continue
            if src_x < 0.0 or src_y < 0.0 or src_x > max_x or src_y > max_y:
                continue

            x0 = int(math.floor(src_x))
            y0 = int(math.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1
            if x1 >= width or y1 >= height:
                continue

            fx = src_x - x0
            fy = src_y - y0
            idx00 = y0 * width + x0
            idx10 = y0 * width + x1
            idx01 = y1 * width + x0
            idx11 = y1 * width + x1

            a = src[idx00]
            b = src[idx10]
            c = src[idx01]
            d = src[idx11]

            w00 = (1.0 - fx) * (1.0 - fy)
            w10 = fx * (1.0 - fy)
            w01 = (1.0 - fx) * fy
            w11 = fx * fy
            value = a * w00 + b * w10 + c * w01 + d * w11

            idx = y * width + x
            out[idx] = clamp_u8(round(value))
            mask[idx] = 1

    return out, mask


def gaussian_kernel_1d(sigma: float) -> List[float]:
    if sigma <= 0.0:
        return [1.0]
    radius = max(1, int(math.ceil(3.0 * sigma)))
    denom = 2.0 * sigma * sigma
    kernel = [math.exp(-(x * x) / denom) for x in range(-radius, radius + 1)]
    norm = sum(kernel)
    return [v / norm for v in kernel]


def gaussian_blur_u8(data: List[int], width: int, height: int, sigma: float) -> List[int]:
    kernel = gaussian_kernel_1d(sigma)
    if len(kernel) == 1:
        return data[:]

    radius = len(kernel) // 2
    temp = [0.0] * (width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            acc = 0.0
            for k, w in enumerate(kernel):
                xx = x + k - radius
                if xx < 0:
                    xx = 0
                elif xx >= width:
                    xx = width - 1
                acc += data[row + xx] * w
            temp[row + x] = acc

    out = [0] * (width * height)
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for k, w in enumerate(kernel):
                yy = y + k - radius
                if yy < 0:
                    yy = 0
                elif yy >= height:
                    yy = height - 1
                acc += temp[yy * width + x] * w
            out[y * width + x] = clamp_u8(round(acc))
    return out


def apply_gain_bias_inplace(data: List[int], gain: float, bias: float) -> None:
    if gain == 1.0 and bias == 0.0:
        return
    for i, value in enumerate(data):
        data[i] = clamp_u8(round(value * gain + bias))


def add_gaussian_noise_inplace(data: List[int], sigma: float, rng: random.Random) -> None:
    if sigma <= 0.0:
        return
    for i, value in enumerate(data):
        data[i] = clamp_u8(round(value + rng.gauss(0.0, sigma)))


def add_salt_pepper_inplace(data: List[int], prob: float, rng: random.Random) -> None:
    if prob <= 0.0:
        return
    for i in range(len(data)):
        if rng.random() < prob:
            data[i] = 0 if rng.random() < 0.5 else 255


def pattern_xor(width: int, height: int) -> List[int]:
    out = []
    for y in range(height):
        for x in range(width):
            value = ((x * 13) ^ (y * 7) ^ (x * y)) & 0xFF
            out.append(value)
    return out


def pattern_checker(width: int, height: int, cell: int) -> List[int]:
    out = []
    for y in range(height):
        for x in range(width):
            on = ((x // cell) + (y // cell)) % 2 == 0
            out.append(32 if on else 224)
    return out


def pattern_rings(width: int, height: int, freq: float) -> List[int]:
    out = []
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            r = math.hypot(dx, dy)
            value = 128.0 + 110.0 * math.sin(r * freq)
            out.append(clamp_u8(round(value)))
    return out


def pattern_bars(width: int, height: int, cell: int) -> List[int]:
    out = []
    for y in range(height):
        for x in range(width):
            on = (x // cell) % 2 == 0
            out.append(40 if on else 210)
    return out


def pattern_noise(width: int, height: int, rng: random.Random) -> List[int]:
    return [rng.randint(0, 255) for _ in range(width * height)]


def pattern_asymmetric(width: int, height: int, rng: random.Random) -> List[int]:
    """
    Generate an asymmetric pattern (L-shape + diagonal gradient) that has no
    rotational symmetry, ideal for testing fine rotation detection.
    """
    out = [80] * (width * height)

    # Horizontal bar in upper quarter
    bar_h = height // 4
    for y in range(bar_h):
        for x in range(width):
            out[y * width + x] = 200

    # Vertical bar on left quarter (creates L-shape)
    bar_w = width // 4
    for y in range(height):
        for x in range(bar_w):
            out[y * width + x] = 200

    # Diagonal gradient in lower-right quadrant (breaks all symmetry)
    for y in range(height // 2, height):
        for x in range(width // 2, width):
            t = (x - width // 2) / max(1, width // 2)
            value = int(40 + 180 * t)
            out[y * width + x] = clamp_u8(value)

    # Add some texture/noise
    for i in range(len(out)):
        jitter = rng.randint(-15, 15)
        out[i] = clamp_u8(out[i] + jitter)

    return out


def make_pattern(
    name: str, width: int, height: int, rng: random.Random
) -> List[int]:
    if name == "xor":
        return pattern_xor(width, height)
    if name == "checker":
        cell = max(2, min(width, height) // 8)
        return pattern_checker(width, height, cell)
    if name == "rings":
        return pattern_rings(width, height, 0.35)
    if name == "bars":
        cell = max(2, width // 12)
        return pattern_bars(width, height, cell)
    if name == "noise":
        data = pattern_noise(width, height, rng)
        return gaussian_blur_u8(data, width, height, 0.6)
    if name == "asymmetric":
        return pattern_asymmetric(width, height, rng)
    raise ValueError(f"unknown pattern '{name}'")


def make_background(
    style: str, width: int, height: int, rng: random.Random, value: Optional[int]
) -> List[int]:
    if style == "flat":
        fill = value if value is not None else rng.randint(20, 200)
        return [fill] * (width * height)
    if style == "gradient":
        base = rng.randint(40, 140)
        ax = rng.uniform(-0.4, 0.4)
        ay = rng.uniform(-0.4, 0.4)
        out = []
        for y in range(height):
            for x in range(width):
                out.append(clamp_u8(round(base + ax * x + ay * y)))
        return out
    if style == "xor":
        out = []
        for y in range(height):
            for x in range(width):
                value = ((x * 9 + y * 5 + x * y) & 0xFF)
                out.append(value)
        return out
    if style == "noise":
        return pattern_noise(width, height, rng)
    if style == "rings":
        return pattern_rings(width, height, 0.22)
    if style == "mixed":
        base = make_background("gradient", width, height, rng, value)
        out = []
        for i, value in enumerate(base):
            jitter = rng.randint(-20, 20)
            out.append(clamp_u8(value + jitter))
        return out
    raise ValueError(f"unknown background '{style}'")


def rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def choose_position(
    rng: random.Random,
    img_w: int,
    img_h: int,
    tpl_w: int,
    tpl_h: int,
    place_mode: str,
    avoid: List[Tuple[int, int, int, int]],
) -> Tuple[int, int]:
    margin = 2
    max_x = max(0, img_w - tpl_w)
    max_y = max(0, img_h - tpl_h)

    def pick() -> Tuple[int, int]:
        if place_mode == "edge":
            edge = rng.choice(["left", "right", "top", "bottom"])
            if edge == "left":
                x0 = rng.randint(0, min(margin, max_x))
                y0 = rng.randint(0, max_y)
            elif edge == "right":
                x0 = rng.randint(max(0, max_x - margin), max_x)
                y0 = rng.randint(0, max_y)
            elif edge == "top":
                x0 = rng.randint(0, max_x)
                y0 = rng.randint(0, min(margin, max_y))
            else:
                x0 = rng.randint(0, max_x)
                y0 = rng.randint(max(0, max_y - margin), max_y)
            return x0, y0
        x0 = rng.randint(margin, max_x - margin) if max_x > margin * 2 else 0
        y0 = rng.randint(margin, max_y - margin) if max_y > margin * 2 else 0
        return x0, y0

    for _ in range(80):
        x0, y0 = pick()
        rect = (x0, y0, tpl_w, tpl_h)
        if not any(rects_overlap(rect, other) for other in avoid):
            return x0, y0
    return pick()


def embed_template(
    image: List[int],
    img_w: int,
    tpl: List[int],
    mask: List[int],
    tpl_w: int,
    tpl_h: int,
    x0: int,
    y0: int,
    gain: float,
    bias: float,
) -> None:
    for y in range(tpl_h):
        row = (y0 + y) * img_w
        tpl_row = y * tpl_w
        for x in range(tpl_w):
            idx = tpl_row + x
            if mask[idx] == 0:
                continue
            value = tpl[idx] * gain + bias
            image[row + x0 + x] = clamp_u8(round(value))


def apply_occlusion(
    image: List[int],
    background: List[int],
    img_w: int,
    img_h: int,
    x0: int,
    y0: int,
    tpl_w: int,
    tpl_h: int,
    frac: float,
    rng: random.Random,
) -> Optional[Dict[str, int]]:
    if frac <= 0.0:
        return None
    area = int(tpl_w * tpl_h * frac)
    area = max(1, min(area, tpl_w * tpl_h))
    ratio = rng.uniform(0.4, 2.5)
    occ_w = max(1, min(tpl_w, int(round(math.sqrt(area) * ratio))))
    occ_h = max(1, min(tpl_h, int(round(area / occ_w))))
    ox = rng.randint(x0, x0 + tpl_w - occ_w)
    oy = rng.randint(y0, y0 + tpl_h - occ_h)

    for y in range(occ_h):
        row = (oy + y) * img_w
        for x in range(occ_w):
            idx = row + ox + x
            image[idx] = background[idx]

    return {"x": ox, "y": oy, "width": occ_w, "height": occ_h}


def stable_seed(base_seed: int, case_id: str, index: int) -> int:
    h = 2166136261
    for ch in case_id:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    h ^= (index + 1) * 0x9E3779B1
    h ^= base_seed & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def save_png(path: Path, data: List[int], width: int, height: int) -> None:
    img = Image.frombytes("L", (width, height), bytes(data))
    img.save(path, format="PNG")


def build_config(
    spec: CaseSpec, rotation_mode: str
) -> Tuple[Dict[str, object], Dict[str, object]]:
    compile_cfg = dict(DEFAULT_COMPILE)
    if spec.compile_overrides:
        compile_cfg.update(spec.compile_overrides)

    match_cfg = dict(DEFAULT_MATCH)
    if spec.match_overrides:
        match_cfg.update(spec.match_overrides)

    match_cfg["rotation"] = rotation_mode
    return compile_cfg, match_cfg


def generate_case(
    spec: CaseSpec,
    out_dir: Path,
    base_seed: int,
    case_index: int,
    emit_mask: bool,
) -> Dict[str, object]:
    case_seed = stable_seed(base_seed, spec.case_id, case_index)
    rng = random.Random(case_seed)

    tpl_w, tpl_h = spec.template_size
    img_w, img_h = spec.image_size
    template = make_pattern(spec.template_pattern, tpl_w, tpl_h, rng)
    background = make_background(
        spec.background_style, img_w, img_h, rng, spec.background_value
    )
    image = background[:]

    instances = []
    avoid = []
    occlusion = None

    rotation_mode = "enabled" if abs(spec.rotation_deg) > 1e-6 else "disabled"
    if spec.match_overrides and "rotation" in spec.match_overrides:
        rotation_mode = str(spec.match_overrides["rotation"])
    rotation_enabled = rotation_mode == "enabled"
    target_mask = None
    if spec.present:
        x0, y0 = choose_position(
            rng, img_w, img_h, tpl_w, tpl_h, spec.place_mode, avoid
        )
        if rotation_enabled:
            rot, mask = rotate_u8_bilinear_masked(
                template, tpl_w, tpl_h, spec.rotation_deg, 0
            )
            target_mask = mask
        else:
            rot = template
            mask = [1] * (tpl_w * tpl_h)
        embed_template(
            image, img_w, rot, mask, tpl_w, tpl_h, x0, y0, spec.template_gain, spec.template_bias
        )
        avoid.append((x0, y0, tpl_w, tpl_h))
        instances.append(
            {
                "kind": "target",
                "x": x0,
                "y": y0,
                "angle_deg": spec.rotation_deg,
                "gain": spec.template_gain,
                "bias": spec.template_bias,
            }
        )
        if spec.occlusion_frac > 0.0:
            occlusion = apply_occlusion(
                image, background, img_w, img_h, x0, y0, tpl_w, tpl_h, spec.occlusion_frac, rng
            )

    for i in range(spec.distractors):
        dx, dy = choose_position(rng, img_w, img_h, tpl_w, tpl_h, "random", avoid)
        angle = 0.0
        if rotation_enabled:
            angle = spec.rotation_deg + rng.uniform(
                -spec.distractor_jitter_deg, spec.distractor_jitter_deg
            )
        if rotation_enabled:
            rot, d_mask = rotate_u8_bilinear_masked(template, tpl_w, tpl_h, angle, 0)
        else:
            rot = template
            d_mask = [1] * (tpl_w * tpl_h)
        gain = spec.template_gain * rng.uniform(0.9, 1.1)
        bias = spec.template_bias + rng.uniform(-5.0, 5.0)
        embed_template(image, img_w, rot, d_mask, tpl_w, tpl_h, dx, dy, gain, bias)
        avoid.append((dx, dy, tpl_w, tpl_h))
        instances.append(
            {"kind": "distractor", "x": dx, "y": dy, "angle_deg": angle, "gain": gain, "bias": bias}
        )

    apply_gain_bias_inplace(image, spec.global_gain, spec.global_bias)
    if spec.blur_sigma > 0.0:
        image = gaussian_blur_u8(image, img_w, img_h, spec.blur_sigma)
    add_gaussian_noise_inplace(image, spec.noise_sigma, rng)
    add_salt_pepper_inplace(image, spec.salt_pepper, rng)

    compile_cfg, match_cfg = build_config(spec, rotation_mode)
    topk = spec.topk
    if topk is None:
        target_count = 1 if spec.present else 0
        topk = max(1, target_count + spec.distractors)

    save_png(out_dir / "image.png", image, img_w, img_h)
    save_png(out_dir / "template.png", template, tpl_w, tpl_h)
    if emit_mask and rotation_enabled and target_mask is not None:
        mask_img = [value * 255 for value in target_mask]
        save_png(out_dir / "mask.png", mask_img, tpl_w, tpl_h)

    cli_config = {
        "image_path": "image.png",
        "template_path": "template.png",
        "topk": topk,
        "compile": compile_cfg,
        "match": match_cfg,
    }
    with (out_dir / "cli_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cli_config, handle, indent=2, sort_keys=True)

    meta = {
        "case_id": spec.case_id,
        "family": spec.family,
        "seed": case_seed,
        "present": spec.present,
        "image": {"width": img_w, "height": img_h},
        "template": {
            "width": tpl_w,
            "height": tpl_h,
            "pattern": spec.template_pattern,
        },
        "background": {
            "style": spec.background_style,
            "value": spec.background_value,
        },
        "rotation_deg": spec.rotation_deg,
        "effects": {
            "template_gain": spec.template_gain,
            "template_bias": spec.template_bias,
            "global_gain": spec.global_gain,
            "global_bias": spec.global_bias,
            "noise_sigma": spec.noise_sigma,
            "blur_sigma": spec.blur_sigma,
            "salt_pepper": spec.salt_pepper,
            "occlusion_frac": spec.occlusion_frac,
        },
        "instances": instances,
        "occlusion": occlusion,
        "cli_config": cli_config,
        "notes": spec.notes,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return {
        "case_id": spec.case_id,
        "family": spec.family,
        "dir": out_dir.name,
        "present": spec.present,
        "image": str(Path(out_dir.name) / "image.png"),
        "template": str(Path(out_dir.name) / "template.png"),
        "meta": str(Path(out_dir.name) / "meta.json"),
        "cli_config": str(Path(out_dir.name) / "cli_config.json"),
        "topk": topk,
    }


def base_cases_standard() -> List[CaseSpec]:
    return [
        CaseSpec(
            case_id="clean_translation",
            family="translation",
            image_size=(256, 192),
            template_size=(64, 48),
            template_pattern="xor",
            background_style="flat",
            rotation_deg=0.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 3},
            compile_overrides={"max_levels": 3},
        ),
        CaseSpec(
            case_id="clean_translation_large",
            family="translation",
            image_size=(1024, 768),
            template_size=(192, 160),
            template_pattern="xor",
            background_style="gradient",
            rotation_deg=0.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 5},
            compile_overrides={"max_levels": 5},
        ),
        CaseSpec(
            case_id="rotation_coarse_30deg",
            family="rotation",
            image_size=(320, 240),
            template_size=(96, 72),
            template_pattern="xor",
            background_style="flat",
            rotation_deg=30.0,
            match_overrides={"rotation": "enabled", "max_image_levels": 3},
            compile_overrides={"max_levels": 3, "coarse_step_deg": 30.0, "min_step_deg": 30.0},
        ),
        CaseSpec(
            case_id="rotation_fine_22_5deg",
            family="rotation",
            image_size=(320, 240),
            template_size=(80, 60),
            template_pattern="asymmetric",  # Changed from rings to avoid 180° symmetry
            background_style="gradient",
            rotation_deg=22.5,
            match_overrides={"rotation": "enabled", "max_image_levels": 4},
            compile_overrides={"max_levels": 4, "coarse_step_deg": 30.0, "min_step_deg": 7.5},
        ),
        CaseSpec(
            case_id="rotation_wrap_172_5deg",
            family="rotation",
            image_size=(300, 220),
            template_size=(72, 56),
            template_pattern="bars",
            background_style="flat",
            rotation_deg=172.5,
            match_overrides={"rotation": "enabled", "max_image_levels": 4},
            compile_overrides={"max_levels": 4, "coarse_step_deg": 30.0, "min_step_deg": 7.5},
        ),
        CaseSpec(
            case_id="noise_gaussian",
            family="noise",
            image_size=(320, 240),
            template_size=(80, 64),
            template_pattern="xor",
            background_style="gradient",
            rotation_deg=0.0,
            noise_sigma=12.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 4},
        ),
        CaseSpec(
            case_id="blur_sigma_1_5",
            family="blur",
            image_size=(320, 240),
            template_size=(88, 68),
            template_pattern="checker",
            background_style="mixed",
            rotation_deg=0.0,
            blur_sigma=1.5,
            match_overrides={"rotation": "disabled", "max_image_levels": 4},
        ),
        CaseSpec(
            case_id="illumination_shift",
            family="illumination",
            image_size=(300, 220),
            template_size=(72, 52),
            template_pattern="xor",
            background_style="gradient",
            rotation_deg=0.0,
            template_gain=1.25,
            template_bias=14.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 3},
        ),
        CaseSpec(
            case_id="occluded_25pct",
            family="occlusion",
            image_size=(320, 240),
            template_size=(90, 70),
            template_pattern="xor",
            background_style="flat",
            rotation_deg=0.0,
            occlusion_frac=0.25,
            noise_sigma=4.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 4},
        ),
        CaseSpec(
            case_id="distractors_topk",
            family="distractors",
            image_size=(420, 320),
            template_size=(80, 60),
            template_pattern="rings",
            background_style="mixed",
            rotation_deg=0.0,
            distractors=3,
            topk=4,
            match_overrides={"rotation": "disabled", "max_image_levels": 4, "per_angle_topk": 4},
        ),
        CaseSpec(
            case_id="near_border",
            family="edge",
            image_size=(320, 240),
            template_size=(88, 68),
            template_pattern="bars",
            background_style="gradient",
            rotation_deg=0.0,
            place_mode="edge",
            match_overrides={"rotation": "disabled", "max_image_levels": 4},
        ),
        CaseSpec(
            case_id="negative_no_match",
            family="negative",
            image_size=(320, 240),
            template_size=(80, 64),
            template_pattern="xor",
            background_style="noise",
            rotation_deg=0.0,
            present=False,
            noise_sigma=6.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 4},
        ),
        CaseSpec(
            case_id="pyramid_stress",
            family="pyramid",
            image_size=(1200, 900),
            template_size=(160, 120),
            template_pattern="checker",
            background_style="xor",
            rotation_deg=0.0,
            noise_sigma=3.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 6},
            compile_overrides={"max_levels": 6},
        ),
        # Single-level test cases (no pyramid) to isolate correlation from pyramid issues
        CaseSpec(
            case_id="pyramid_stress_single_level",
            family="pyramid",
            image_size=(1200, 900),
            template_size=(160, 120),
            template_pattern="checker",
            background_style="xor",
            rotation_deg=0.0,
            noise_sigma=3.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 1},
            compile_overrides={"max_levels": 1},
        ),
        CaseSpec(
            case_id="rotation_fine_single_level",
            family="rotation",
            image_size=(320, 240),
            template_size=(80, 60),
            template_pattern="asymmetric",
            background_style="gradient",
            rotation_deg=22.5,
            match_overrides={"rotation": "enabled", "max_image_levels": 1},
            compile_overrides={"max_levels": 1, "coarse_step_deg": 5.0, "min_step_deg": 1.0},
        ),
    ]


def base_cases_smoke() -> List[CaseSpec]:
    return [
        CaseSpec(
            case_id="smoke_translation",
            family="translation",
            image_size=(192, 144),
            template_size=(56, 40),
            template_pattern="xor",
            background_style="flat",
            rotation_deg=0.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 3},
            compile_overrides={"max_levels": 3},
        ),
        CaseSpec(
            case_id="smoke_rotation",
            family="rotation",
            image_size=(256, 192),
            template_size=(64, 48),
            template_pattern="rings",
            background_style="gradient",
            rotation_deg=22.5,
            match_overrides={"rotation": "enabled", "max_image_levels": 3},
            compile_overrides={"max_levels": 3, "coarse_step_deg": 30.0, "min_step_deg": 7.5},
        ),
        CaseSpec(
            case_id="smoke_negative",
            family="negative",
            image_size=(192, 144),
            template_size=(56, 40),
            template_pattern="bars",
            background_style="noise",
            rotation_deg=0.0,
            present=False,
            match_overrides={"rotation": "disabled", "max_image_levels": 3},
        ),
    ]


def base_cases_performance() -> List[CaseSpec]:
    return [
        CaseSpec(
            case_id="perf_large_translation",
            family="performance",
            image_size=(1600, 1200),
            template_size=(240, 180),
            template_pattern="xor",
            background_style="xor",
            rotation_deg=0.0,
            match_overrides={"rotation": "disabled", "max_image_levels": 6},
            compile_overrides={"max_levels": 6},
        ),
        CaseSpec(
            case_id="perf_large_rotation",
            family="performance",
            image_size=(1400, 1000),
            template_size=(220, 160),
            template_pattern="rings",
            background_style="mixed",
            rotation_deg=22.5,
            match_overrides={"rotation": "enabled", "max_image_levels": 6},
            compile_overrides={"max_levels": 6, "coarse_step_deg": 30.0, "min_step_deg": 7.5},
        ),
        CaseSpec(
            case_id="perf_many_distractors",
            family="performance",
            image_size=(1024, 1024),
            template_size=(128, 128),
            template_pattern="checker",
            background_style="mixed",
            rotation_deg=0.0,
            distractors=8,
            topk=6,
            match_overrides={"rotation": "disabled", "max_image_levels": 5, "per_angle_topk": 6},
            compile_overrides={"max_levels": 5},
        ),
        CaseSpec(
            case_id="perf_noise_blur",
            family="performance",
            image_size=(1200, 900),
            template_size=(160, 120),
            template_pattern="xor",
            background_style="gradient",
            rotation_deg=0.0,
            noise_sigma=10.0,
            blur_sigma=1.2,
            match_overrides={"rotation": "disabled", "max_image_levels": 6},
            compile_overrides={"max_levels": 6},
        ),
    ]


def build_suite(name: str) -> List[CaseSpec]:
    if name == "smoke":
        return base_cases_smoke()
    if name == "standard":
        return base_cases_standard()
    if name == "performance":
        return base_cases_performance()
    raise ValueError(f"unknown suite '{name}'")


def expand_cases(cases: List[CaseSpec], count: int) -> List[CaseSpec]:
    if count <= 1:
        return cases
    expanded = []
    for spec in cases:
        for idx in range(count):
            suffix = f"_{idx + 1}"
            expanded.append(replace(spec, case_id=f"{spec.case_id}{suffix}"))
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PNG cases for CorrMatch.",
    )
    parser.add_argument("--out", type=Path, default=Path("synthetic_cases"))
    parser.add_argument("--suite", choices=["smoke", "standard", "performance"], default="standard")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--emit-mask", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    args = parser.parse_args()

    cases = expand_cases(build_suite(args.suite), args.cases_per_family)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.case_id in wanted]
    if args.family:
        wanted = set(args.family)
        cases = [case for case in cases if case.family in wanted]

    if args.list:
        for case in cases:
            print(case.case_id)
        return 0

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    for index, spec in enumerate(cases):
        case_dir = out_dir / spec.case_id
        if case_dir.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"{case_dir} already exists. Use --overwrite or choose a new --out."
                )
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest_cases.append(
            generate_case(spec, case_dir, args.seed, index, emit_mask=args.emit_mask)
        )

    manifest = {
        "suite": args.suite,
        "seed": args.seed,
        "cases_per_family": args.cases_per_family,
        "cases": manifest_cases,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
