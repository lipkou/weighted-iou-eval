"""Regression test on road_data.

Expected values were computed by an independent one-off script (greedy
matching, brute-force point-in-polygon on pixel centers with floored
vertices, plain sums), not by this package. The package must reproduce them.
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
    "0.1": ("0.3", 0.869971), "0.2": ("0.9", 0.729730), "0.3": ("0.2", 0.735786),
    "0.5": ("0.7", 0.397024), "0.6": ("0.6", 0.808989),
    "0.7": ("0.8", 0.508884), "0.8": ("0.1", 0.434938),
}

# (object_weight, fp_penalty) -> (car, road_sign, road_barrier, file), in %
EXPECTED_SCORES = {
    ("area", 0.0): (79.6531, 43.5646, 46.4504, 59.8826),
    ("area", 0.5): (79.6531, 40.2349, 46.4504, 58.7244),
    ("area", 1.0): (79.6531, 37.3781, 46.4504, 57.7308),
    ("sqrt_area", 0.0): (72.0036, 49.7280, 46.8169, 58.7802),
    ("sqrt_area", 0.5): (72.0036, 42.8071, 46.8169, 56.3729),
    ("sqrt_area", 1.0): (72.0036, 37.5773, 46.8169, 54.5539),
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
    # GT 0.7 <-> pred 0.8 has IoU 0.509: just above the threshold
    assert (c["road_barrier"].precision, c["road_barrier"].recall) == (0.5, 0.5)


def test_per_class_tp_threshold(gt, pred, cfg):
    classes = dict(cfg.classes)
    classes["road_barrier"] = replace(classes["road_barrier"], tp_iou=0.3)
    res = evaluate(gt, pred, replace(cfg, classes=classes))
    b = res.classes["road_barrier"]
    assert (b.precision, b.recall) == (1.0, 1.0)
    # the threshold changes only the counts, never the score
    assert b.score * 100 == pytest.approx(46.4504, abs=1e-4)


def test_attributes(gt, pred, cfg):
    a = evaluate(gt, pred, cfg).attributes
    assert (a.agreed, a.compared) == (1, 3)
    assert {(m.gt_nm, m.pred_nm, m.gt_value, m.pred_value) for m in a.mismatches} == {
        ("0.1", "0.3", False, None), ("0.3", "0.2", False, True),
    }
    assert [(m.key, m.gt_value, m.pred_value) for m in a.frame_mismatches] == [("weather", "clear", "fog")]
