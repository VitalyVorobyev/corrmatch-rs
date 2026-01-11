"""Tests using synthetic test cases to validate corrmatch Python bindings."""

import json

import numpy as np
import pytest

from conftest import (
    ANGLE_TOLERANCE_DEG,
    MIN_SCORE_THRESHOLD,
    POSITION_TOLERANCE_PX,
    SyntheticCase,
    assert_match_close,
    load_image,
)

# Skip all tests if corrmatch is not available
pytest.importorskip("corrmatch")
import corrmatch


class TestSyntheticCases:
    """Test corrmatch against synthetic test cases with ground truth."""

    def test_match_present(self, synthetic_case: SyntheticCase):
        """Test that matcher finds template when present."""
        if not synthetic_case.expected_present:
            pytest.skip("Case expects no match")

        if not synthetic_case.instances:
            pytest.skip("No ground truth instances")

        image = load_image(synthetic_case.image_path)
        template = load_image(synthetic_case.template_path)

        # Load CLI config to determine rotation mode
        with open(synthetic_case.cli_config_path) as f:
            cli_config = json.load(f)

        rotation = cli_config.get("match", {}).get("rotation", "disabled")
        metric = cli_config.get("match", {}).get("metric", "zncc")

        # Run matcher
        result = corrmatch.match_template(
            image, template,
            rotation=rotation,
            metric=metric,
        )

        # Get primary expected instance
        expected = synthetic_case.instances[0]

        # Validate position and angle
        assert_match_close(result, expected)

        # Check score is reasonable
        if metric == "zncc":
            assert result.score >= MIN_SCORE_THRESHOLD, \
                f"Score {result.score:.4f} below threshold {MIN_SCORE_THRESHOLD}"

    def test_match_absent(self, synthetic_case: SyntheticCase):
        """Test that matcher doesn't find template when absent."""
        if synthetic_case.expected_present:
            pytest.skip("Case expects match present")

        image = load_image(synthetic_case.image_path)
        template = load_image(synthetic_case.template_path)

        # Run matcher
        result = corrmatch.match_template(image, template)

        # Score should be low for absent template
        # (This is a weak check - we mainly verify it doesn't crash)
        assert result.score < MIN_SCORE_THRESHOLD, \
            f"Unexpectedly high score {result.score:.4f} for absent template"


class TestBasicFunctionality:
    """Basic functionality tests that don't require synthetic cases."""

    def test_template_creation(self):
        """Test creating a template from numpy array."""
        data = np.random.randint(0, 256, (32, 48), dtype=np.uint8)
        tpl = corrmatch.Template(data)
        assert tpl.width == 48
        assert tpl.height == 32

    def test_compile_with_rotation(self):
        """Test compiling template with rotation support."""
        data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        tpl = corrmatch.Template(data)
        compiled = tpl.compile()
        assert compiled.num_levels > 0

    def test_compile_no_rotation(self):
        """Test compiling template without rotation."""
        data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        tpl = corrmatch.Template(data)
        compiled = tpl.compile_no_rotation()
        assert compiled.num_levels > 0

    def test_simple_match(self):
        """Test matching a template against an image with exact copy."""
        # Create image with embedded template
        image = np.zeros((128, 128), dtype=np.uint8)
        template = np.random.randint(50, 200, (32, 32), dtype=np.uint8)

        # Place template at known position
        x, y = 40, 60
        image[y:y+32, x:x+32] = template

        # Run matcher
        result = corrmatch.match_template(image, template)

        # Check position (should be exact for copy)
        assert abs(result.x - x) <= 1.0, f"x error: got {result.x}, expected {x}"
        assert abs(result.y - y) <= 1.0, f"y error: got {result.y}, expected {y}"
        assert result.score > 0.95, f"Score too low: {result.score}"

    def test_topk_matches(self):
        """Test returning multiple matches."""
        # Create image with two template copies
        image = np.zeros((128, 128), dtype=np.uint8)
        template = np.random.randint(50, 200, (24, 24), dtype=np.uint8)

        # Place template at two positions
        image[20:44, 20:44] = template
        image[80:104, 80:104] = template

        # Compile and match
        tpl = corrmatch.Template(template)
        compiled = tpl.compile_no_rotation(max_levels=4)
        matcher = compiled.matcher()
        results = matcher.match_topk(image, k=2)

        assert len(results) == 2
        # Both should have high scores
        for r in results:
            assert r.score > 0.9

    def test_config_validation(self):
        """Test that invalid configs are rejected."""
        with pytest.raises(RuntimeError):
            corrmatch.MatchConfig(beam_width=0)

        with pytest.raises(RuntimeError):
            corrmatch.MatchConfig(per_angle_topk=0)

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError):
            corrmatch.MatchConfig(metric="invalid")

    def test_invalid_rotation_mode(self):
        """Test that invalid rotation mode raises error."""
        with pytest.raises(ValueError):
            corrmatch.MatchConfig(rotation="invalid")


class TestRotation:
    """Tests specifically for rotation matching."""

    def test_rotated_template_detection(self):
        """Test detecting a rotated template."""
        # Create a simple asymmetric template
        template = np.zeros((32, 32), dtype=np.uint8)
        template[8:24, 8:12] = 200  # Vertical bar

        # Create rotated version (90 degrees = horizontal bar)
        template_rot = np.zeros((32, 32), dtype=np.uint8)
        template_rot[8:12, 8:24] = 200  # Horizontal bar

        # Place rotated template in image
        image = np.zeros((96, 96), dtype=np.uint8)
        x, y = 32, 32
        image[y:y+32, x:x+32] = template_rot

        # Match with rotation enabled
        result = corrmatch.match_template(
            image, template,
            rotation="enabled",
        )

        # Should find template near correct position
        assert abs(result.x - x) <= 4.0
        assert abs(result.y - y) <= 4.0
        # Angle should be near 90 or -90 degrees
        angle_error = min(abs(result.angle_deg - 90), abs(result.angle_deg + 90))
        assert angle_error <= 15.0, f"Angle error {angle_error} too large"


class TestParallel:
    """Tests for parallel execution."""

    def test_parallel_matches_sequential(self):
        """Test that parallel execution gives same results as sequential."""
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        template = np.random.randint(50, 200, (48, 48), dtype=np.uint8)

        # Place template
        image[100:148, 100:148] = template

        # Match sequentially
        seq_result = corrmatch.match_template(
            image, template,
            parallel=False,
        )

        # Match in parallel
        par_result = corrmatch.match_template(
            image, template,
            parallel=True,
        )

        # Results should be very close
        assert abs(seq_result.x - par_result.x) <= 0.1
        assert abs(seq_result.y - par_result.y) <= 0.1
        assert abs(seq_result.score - par_result.score) <= 0.001
