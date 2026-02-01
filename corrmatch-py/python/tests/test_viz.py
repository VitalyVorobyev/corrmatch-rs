import numpy as np
import pytest


pytest.importorskip("corrmatch")
import corrmatch  # noqa: E402


def test_config_getters_smoke():
    cfg = corrmatch.MatchConfig(metric="zncc", rotation="enabled", beam_width=7)
    assert cfg.metric == "zncc"
    assert cfg.rotation == "enabled"
    assert cfg.beam_width == 7

    ccfg = corrmatch.CompileConfig(max_levels=5, coarse_step_deg=12.0)
    assert ccfg.max_levels == 5
    assert abs(ccfg.coarse_step_deg - 12.0) < 1e-6


def test_rotate_u8_bilinear_masked_zero_deg():
    img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    rot, mask = corrmatch.rotate_u8_bilinear_masked(img, angle_deg=0.0, fill_value=0)

    assert rot.shape == img.shape
    assert mask.shape == img.shape

    # For 0 deg rotation, interior pixels should match exactly (mask==1).
    valid = mask != 0
    assert np.array_equal(rot[valid], img[valid])


def test_viz_smoke_no_show():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    import corrmatch.viz as viz  # noqa: E402

    image = np.zeros((64, 64), dtype=np.uint8)
    template = np.random.randint(50, 200, (16, 16), dtype=np.uint8)
    image[20:36, 30:46] = template

    match_cfg = corrmatch.MatchConfig(rotation="disabled", metric="zncc")
    _fig, matches = viz.match_and_show(
        image, template, topk=3, match_cfg=match_cfg, show=False, title="test"
    )
    assert len(matches) == 3


def test_viz_rotation_direction_cw():
    import corrmatch.viz as viz  # noqa: E402

    v = np.array([[1.0, 0.0]], dtype=np.float32)
    out = viz._rot_cw(v, 90.0)
    assert np.allclose(out, np.array([[0.0, 1.0]], dtype=np.float32), atol=1e-5)
