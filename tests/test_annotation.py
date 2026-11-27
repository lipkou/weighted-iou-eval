import json
import math
from pathlib import Path

import numpy as np
import pytest

from weighted_iou.annotation import load_annotation, normalize_class, rasterize

DATA = Path(__file__).resolve().parents[1] / "road_data"

# Areas in pixels, from the one-off exploration script (PIL rasterization).
GT_AREAS = {
    "0.1": ("car", 1841), "0.2": ("car", 503), "0.3": ("car", 440), "0.4": ("car", 94),
    "0.5": ("road_sign", 783), "0.6": ("road_sign", 81),
    "0.7": ("road_barrier", 2377), "0.8": ("road_barrier", 3568),
}
PRED_AREAS = {
    "0.1": ("road_barrier", 7702), "0.2": ("car", 598), "0.3": ("car", 1970),
    "0.5": ("road_sign", 143), "0.6": ("road_sign", 80), "0.7": ("road_sign", 1564),
    "0.8": ("road_barrier", 4502), "0.9": ("car", 393),
}


@pytest.mark.parametrize("raw,expected", [
    ("road sign", "road_sign"), ("Road-Sign", "road_sign"), ("road_barrier", "road_barrier"),
])
def test_normalize_class(raw, expected):
    assert normalize_class(raw) == expected


@pytest.mark.parametrize("sub,expected", [("ground_truth", GT_AREAS), ("annotated", PRED_AREAS)])
def test_road_areas(sub, expected):
    ann = load_annotation(DATA / sub / "road.jpg.json")
    assert (ann.width, ann.height) == (1000, 667)
    assert {o.nm: (o.cls, o.area) for o in ann.objects} == expected


def _gt_geometry():
    blocks = json.loads((DATA / "ground_truth" / "road.jpg.json").read_text(encoding="utf-8"))
    shapes = {o["nm"]: o["shape"] for o in blocks[0]["objects"]}
    masks = {o.nm: o.mask for o in load_annotation(DATA / "ground_truth" / "road.jpg.json").objects}
    return [(o, shapes[o["nm"]], masks[o["nm"]]) for o in blocks[1]["objects"]]


def test_rasterize_reproduces_tool_bbox_masks():
    """The tool's bBox masks are pixel-exact copies of our rasterized JSON bbox."""
    boxes = [(geom, mask) for geom, shape, mask in _gt_geometry() if shape == "bBox"]
    assert len(boxes) == 2
    for geom, mask in boxes:
        assert np.array_equal(rasterize(geom, 1000, 667), mask), geom["nm"]


def test_rasterize_matches_tool_extents_for_all_objects():
    """For polygons only the bbox is in JSON: the mask extent must be floor(x1)..floor(x2)-1."""
    for geom, _, mask in _gt_geometry():
        ys, xs = np.nonzero(mask)
        expected = (math.floor(geom["x1"]), math.floor(geom["y1"]),
                    math.floor(geom["x2"]) - 1, math.floor(geom["y2"]) - 1)
        assert (xs.min(), ys.min(), xs.max(), ys.max()) == expected, geom["nm"]


def test_rasterize_polygon_pixel_centers():
    # Right triangle (0,0)-(4,0)-(0,4): centers with x + y < 4 are inside.
    m = rasterize({"path": [[[0.7, 0.2], [4.9, 0.0], [0.0, 4.5]]]}, 6, 6)
    expected = np.array([[x + 0.5 + y + 0.5 < 4 for x in range(6)] for y in range(6)])
    assert np.array_equal(m, expected)


def test_frame_attributes():
    assert load_annotation(DATA / "ground_truth" / "road.jpg.json").attributes == {"weather": "clear"}
    assert load_annotation(DATA / "annotated" / "road.jpg.json").attributes == {"weather": "fog"}
