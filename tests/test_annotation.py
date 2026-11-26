from pathlib import Path

import pytest

from weighted_iou.annotation import load_annotation, normalize_class

DATA = Path(__file__).resolve().parents[1] / "road_data"

# Areas in pixels, from the one-off exploration script (PIL rasterization).
GT_AREAS = {
    "0.1": ("car", 1841), "0.2": ("car", 503), "0.3": ("car", 440), "0.4": ("car", 94),
    "0.5": ("road_sign", 783), "0.6": ("road_sign", 81),
    "0.7": ("road_barrier", 2377), "0.8": ("road_barrier", 3568),
}
PRED_AREAS = {
    "0.1": ("road_barrier", 7793), "0.2": ("car", 631), "0.3": ("car", 2032),
    "0.5": ("road_sign", 168), "0.6": ("road_sign", 99), "0.7": ("road_sign", 1656),
    "0.8": ("road_barrier", 4609), "0.9": ("car", 428),
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


def test_frame_attributes():
    assert load_annotation(DATA / "ground_truth" / "road.jpg.json").attributes == {"weather": "clear"}
    assert load_annotation(DATA / "annotated" / "road.jpg.json").attributes == {"weather": "fog"}
