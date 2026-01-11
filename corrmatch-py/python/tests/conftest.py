"""Pytest fixtures for corrmatch testing."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pytest

# Path to synthetic test cases (relative to repo root)
SYNTHETIC_CASES_DIR = Path(__file__).parent.parent.parent.parent.parent / "synthetic_cases"


@dataclass
class Instance:
    """Ground truth for a template instance."""
    x: float
    y: float
    angle_deg: float
    gain: float
    bias: float


@dataclass
class SyntheticCase:
    """A synthetic test case with ground truth."""
    case_id: str
    family: str
    image_path: Path
    template_path: Path
    meta_path: Path
    cli_config_path: Path
    instances: List[Instance]
    expected_present: bool
    image_size: tuple[int, int]
    template_size: tuple[int, int]


def load_case(case_dir: Path) -> SyntheticCase:
    """Load a synthetic test case from a directory."""
    meta_path = case_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    instances = []
    for inst in meta.get("instances", []):
        instances.append(Instance(
            x=inst["x"],
            y=inst["y"],
            angle_deg=inst.get("angle_deg", 0.0),
            gain=inst.get("gain", 1.0),
            bias=inst.get("bias", 0.0),
        ))

    return SyntheticCase(
        case_id=meta["case_id"],
        family=meta["family"],
        image_path=case_dir / "image.png",
        template_path=case_dir / "template.png",
        meta_path=meta_path,
        cli_config_path=case_dir / "cli_config.json",
        instances=instances,
        expected_present=meta.get("present", True),
        image_size=(meta["image"]["width"], meta["image"]["height"]),
        template_size=(meta["template"]["width"], meta["template"]["height"]),
    )


def discover_cases() -> List[SyntheticCase]:
    """Discover all synthetic test cases."""
    if not SYNTHETIC_CASES_DIR.exists():
        return []

    cases = []
    manifest_path = SYNTHETIC_CASES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for case_id in manifest.get("cases", []):
            case_dir = SYNTHETIC_CASES_DIR / case_id
            if case_dir.is_dir():
                try:
                    cases.append(load_case(case_dir))
                except Exception as e:
                    print(f"Warning: Failed to load case {case_id}: {e}")
    else:
        # Fallback: discover all subdirectories
        for case_dir in SYNTHETIC_CASES_DIR.iterdir():
            if case_dir.is_dir() and (case_dir / "meta.json").exists():
                try:
                    cases.append(load_case(case_dir))
                except Exception as e:
                    print(f"Warning: Failed to load case {case_dir.name}: {e}")

    return cases


@pytest.fixture(scope="session")
def synthetic_cases() -> List[SyntheticCase]:
    """All available synthetic test cases."""
    return discover_cases()


@pytest.fixture(params=discover_cases(), ids=lambda c: c.case_id)
def synthetic_case(request) -> SyntheticCase:
    """Parametrized fixture for each synthetic test case."""
    return request.param


def load_image(path: Path) -> np.ndarray:
    """Load a grayscale image as numpy array."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not available")

    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.uint8)


# Tolerance settings for validation
POSITION_TOLERANCE_PX = 3.0  # pixels
ANGLE_TOLERANCE_DEG = 2.0    # degrees
MIN_SCORE_THRESHOLD = 0.8    # ZNCC score


def assert_match_close(result, expected: Instance, pos_tol: float = POSITION_TOLERANCE_PX,
                       angle_tol: float = ANGLE_TOLERANCE_DEG):
    """Assert that a match result is close to expected ground truth."""
    dx = abs(result.x - expected.x)
    dy = abs(result.y - expected.y)

    # Wrap angle difference to [-180, 180]
    da = abs(result.angle_deg - expected.angle_deg)
    if da > 180:
        da = 360 - da

    assert dx <= pos_tol, f"x error: {dx:.2f} > {pos_tol} (got {result.x:.2f}, expected {expected.x:.2f})"
    assert dy <= pos_tol, f"y error: {dy:.2f} > {pos_tol} (got {result.y:.2f}, expected {expected.y:.2f})"
    assert da <= angle_tol, f"angle error: {da:.2f} > {angle_tol} (got {result.angle_deg:.2f}, expected {expected.angle_deg:.2f})"
