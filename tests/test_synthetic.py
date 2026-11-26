"""Cases the road data does not contain: wrong class, FP-only class, empty frame."""

import numpy as np
import pytest

from weighted_iou.annotation import AnnotatedObject, Annotation
from weighted_iou.config import ClassSettings, Config
from weighted_iou.metrics import MISSED, UNMATCHED, WRONG_CLASS, evaluate

W, H = 20, 10


def box(x1, y1, x2, y2):
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def ann(*objs):
    return Annotation("t.jpg", W, H, {}, [AnnotatedObject(nm, cls, "bBox", m) for nm, cls, m in objs])


CFG = Config(classes={"car": ClassSettings(1.0, 0.5), "truck": ClassSettings(0.5, 0.5)})


def test_empty_frame_is_perfect():
    res = evaluate(ann(), ann(), CFG)
    assert res.score == 1.0 and res.classes == {}


def test_perfect_prediction():
    res = evaluate(ann(("a", "car", box(0, 0, 4, 4))), ann(("x", "car", box(0, 0, 4, 4))), CFG)
    assert res.score == 1.0


def test_wrong_class_is_miss_and_fp():
    gt = ann(("a", "car", box(0, 0, 4, 4)))
    pred = ann(("x", "truck", box(0, 0, 4, 4)))
    res = evaluate(gt, pred, CFG)
    (o,) = res.objects
    assert (o.status, o.pred_nm, o.pred_cls, o.iou, o.score) == (WRONG_CLASS, "x", "truck", 1.0, 0.0)
    (fp,) = res.false_positives
    assert (fp.reason, fp.cls, fp.gt_nm, fp.gt_cls) == (WRONG_CLASS, "truck", "a", "car")
    assert res.classes["car"].score == 0.0
    assert res.classes["truck"].score == 0.0  # FP-only class is in the total with 0
    assert res.classes["truck"].precision == 0.0 and res.classes["truck"].recall is None
    assert res.score == 0.0


def test_fp_only_class_pulls_total_down():
    gt = ann(("a", "car", box(0, 0, 4, 4)))
    pred = ann(("x", "car", box(0, 0, 4, 4)), ("y", "truck", box(10, 0, 14, 4)))
    res = evaluate(gt, pred, CFG)
    assert res.classes["car"].score == 1.0
    assert res.classes["truck"].score == 0.0
    assert res.score == pytest.approx(1.0 / 1.5)


def test_fp_penalty_in_denominator():
    gt = ann(("a", "car", box(0, 0, 4, 4)))  # area 16
    pred = ann(("x", "car", box(0, 0, 4, 4)), ("y", "car", box(10, 0, 12, 4)))  # FP area 8
    for lam, expected in [(0.0, 1.0), (0.5, 16 / 20), (1.0, 16 / 24)]:
        cfg = Config(fp_penalty=lam, classes=CFG.classes)
        assert evaluate(gt, pred, cfg).classes["car"].score == pytest.approx(expected)


def test_below_min_iou_is_not_a_match():
    gt = ann(("a", "car", box(0, 0, 10, 10)))
    pred = ann(("x", "car", box(9, 0, 19, 10)))  # IoU = 10/190
    res = evaluate(gt, pred, CFG)
    assert res.objects[0].status == MISSED
    assert res.false_positives[0].reason == UNMATCHED


def test_matching_maximizes_total_iou():
    # Greedy takes a-x (0.67) first and is left with b-y (0.14): total 0.81.
    # The optimum is a-y (0.6) + b-x (0.67) = 1.27.
    gt = ann(("a", "car", box(0, 0, 10, 10)), ("b", "car", box(4, 0, 14, 10)))
    pred = ann(("x", "car", box(2, 0, 12, 10)), ("y", "car", box(0, 0, 6, 10)))
    res = evaluate(gt, pred, CFG)
    assert {o.gt_nm: o.pred_nm for o in res.objects} == {"a": "y", "b": "x"}


def test_area_weighting_inside_class():
    gt = ann(("big", "car", box(0, 0, 10, 10)), ("small", "car", box(12, 0, 14, 2)))
    pred = ann(("x", "car", box(0, 0, 10, 10)))  # big found perfectly, small missed
    res = evaluate(gt, pred, CFG)
    assert res.classes["car"].score == pytest.approx(100 / 104)
    assert res.classes["car"].recall == 0.5


def test_unknown_class_fails_loudly():
    with pytest.raises(KeyError, match="bus"):
        evaluate(ann(("a", "bus", box(0, 0, 4, 4))), ann(), CFG)
