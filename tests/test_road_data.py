"""Regression test on road_data.

Expected values were computed by an independent one-off script during the
metric design (greedy matching, PIL rasterization, plain Python sums), before
this package existed. The package must reproduce them.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from weighted_iou.annotation import load_annotation
from weighted_iou.config import load_config
from weighted_iou.metrics import MATCHED, MISSED, UNMATCHED, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "road_data"

# GT nm -> (pred nm, IoU)
EXPECTED_MATCHES = {
    "0.1": ("0.3", 0.855774), "0.2": ("0.9", 0.753296), "0.3": ("0.2", 0.697306),
    "0.5": ("0.7", 0.399312), "0.6": ("0.6", 0.666667),
    "0.7": ("0.8", 0.493054), "0.8": ("0.1", 0.397589),
}

# (object_weight, fp_penalty) -> (car, road_sign, road_barrier, file), in %
EXPECTED_SCORES = {
    ("area", 0.0): (78.5685, 42.4376, 43.5759, 58.3942),
    ("area", 0.5): (78.5685, 38.6773, 43.5759, 57.0862),
    ("area", 1.0): (78.5685, 35.5292, 43.5759, 55.9912),
    ("sqrt_area", 0.0): (71.0789, 46.4375, 44.0491, 56.6319),
    ("sqrt_area", 0.5): (71.0789, 39.5132, 44.0491, 54.2235),
    ("sqrt_area", 1.0): (71.0789, 34.3860, 44.0491, 52.4401),
}


@pytest.fixture(scope="module")
def gt():
    return load_annotation(DATA / "ground_truth" / "road.jpg.json")


@pytest.fixture(scope="module")
def pred():
    return load_annotation(DATA / "annotated" / "road.jpg.json")


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.toml")


def test_matching(gt, pred, cfg):
    res = evaluate(gt, pred, cfg)
    got = {o.gt_nm: (o.pred_nm, o.iou) for o in res.objects if o.status == MATCHED}
    assert got.keys() == EXPECTED_MATCHES.keys()
    for nm, (pnm, iou) in EXPECTED_MATCHES.items():
        assert got[nm][0] == pnm
        assert got[nm][1] == pytest.approx(iou, abs=1e-6)
    assert [o.gt_nm for o in res.objects if o.status == MISSED] == ["0.4"]
    assert [(f.pred_nm, f.cls, f.reason) for f in res.false_positives] == [("0.5", "road_sign", UNMATCHED)]


@pytest.mark.parametrize("key", list(EXPECTED_SCORES))
def test_scores(gt, pred, cfg, key):
    weight, lam = key
    res = evaluate(gt, pred, replace(cfg, object_weight=weight, fp_penalty=lam))
    car, sign, barrier, total = EXPECTED_SCORES[key]
    assert res.classes["car"].score * 100 == pytest.approx(car, abs=1e-4)
    assert res.classes["road_sign"].score * 100 == pytest.approx(sign, abs=1e-4)
    assert res.classes["road_barrier"].score * 100 == pytest.approx(barrier, abs=1e-4)
    assert res.score * 100 == pytest.approx(total, abs=1e-4)


def test_precision_recall_default_threshold(gt, pred, cfg):
    c = evaluate(gt, pred, cfg).classes
    assert (c["car"].tp, c["car"].n_pred, c["car"].n_gt) == (3, 3, 4)
    assert (c["car"].precision, c["car"].recall) == (1.0, 0.75)
    assert (c["road_sign"].tp, c["road_sign"].n_pred, c["road_sign"].n_gt) == (1, 3, 2)
    assert c["road_sign"].recall == 0.5
    assert (c["road_barrier"].precision, c["road_barrier"].recall) == (0.0, 0.0)


def test_per_class_tp_threshold(gt, pred, cfg):
    classes = dict(cfg.classes)
    classes["road_barrier"] = replace(classes["road_barrier"], tp_iou=0.3)
    res = evaluate(gt, pred, replace(cfg, classes=classes))
    b = res.classes["road_barrier"]
    assert (b.precision, b.recall) == (1.0, 1.0)
    # the threshold changes only the counts, never the score
    assert b.score * 100 == pytest.approx(43.5759, abs=1e-4)


def test_attributes(gt, pred, cfg):
    a = evaluate(gt, pred, cfg).attributes
    assert (a.agreed, a.compared) == (1, 3)
    assert {(m.gt_nm, m.pred_nm, m.gt_value, m.pred_value) for m in a.mismatches} == {
        ("0.1", "0.3", False, None), ("0.3", "0.2", False, True),
    }
    assert [(m.key, m.gt_value, m.pred_value) for m in a.frame_mismatches] == [("weather", "clear", "fog")]
