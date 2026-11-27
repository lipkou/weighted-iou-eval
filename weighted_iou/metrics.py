"""The metric.

Object g (ground truth):
    s_g = IoU(g, p) if g is matched to p and cls(p) == cls(g), else 0

Class c (classes of GT and prediction together):
    S_c = sum_g w(A_g) * s_g / ( sum_g w(A_g) + lambda * sum_{p in FP_c} w(A_p) )
    w = area or sqrt(area) (config), lambda = fp_penalty (config).
    FP_c: predictions of class c that are unmatched or matched to a GT object
    of another class. A class with predictions only scores 0.

File:
    S = sum_c W_c * S_c / sum_c W_c   (W_c = class weight from config)
    No objects on either side: S = 1.

Precision / recall per class: TP = matched, same class, IoU >= tp_iou of the class.
Attributes do not enter the score; they are compared on same-class matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotation import Annotation
from .config import Config
from .matching import iou_matrix, match

MATCHED, WRONG_CLASS, MISSED = "matched", "wrong_class", "missed"
UNMATCHED = "unmatched"


@dataclass
class ObjectResult:
    gt_nm: str
    gt_cls: str
    area: int
    status: str  # matched | wrong_class | missed
    pred_nm: str | None = None
    pred_cls: str | None = None
    iou: float = 0.0
    score: float = 0.0
    tp: bool = False


@dataclass
class FalsePositive:
    pred_nm: str
    cls: str
    area: int
    reason: str  # unmatched | wrong_class
    gt_nm: str | None = None  # the GT object it overlapped, for wrong_class
    gt_cls: str | None = None
    iou: float = 0.0


@dataclass
class ClassResult:
    cls: str
    weight: float
    tp_iou: float
    n_gt: int
    n_pred: int
    tp: int
    score: float
    precision: float | None  # None when the class has no predictions
    recall: float | None  # None when the class has no GT objects
    gt_weight: float
    fp_weight: float


@dataclass
class AttributeMismatch:
    gt_nm: str | None
    pred_nm: str | None
    key: str
    gt_value: object
    pred_value: object


@dataclass
class AttributeResult:
    compared: int = 0
    agreed: int = 0
    mismatches: list[AttributeMismatch] = field(default_factory=list)
    frame_mismatches: list[AttributeMismatch] = field(default_factory=list)

    @property
    def agreement(self) -> float | None:
        return self.agreed / self.compared if self.compared else None


@dataclass
class FileResult:
    score: float
    classes: dict[str, ClassResult]
    objects: list[ObjectResult]
    false_positives: list[FalsePositive]
    attributes: AttributeResult


def _compare_attrs(gt: dict, pred: dict, gt_nm, pred_nm, res: AttributeResult, frame=False):
    for key in sorted(set(gt) | set(pred)):
        gv, pv = gt.get(key), pred.get(key)
        mism = AttributeMismatch(gt_nm, pred_nm, key, gv, pv)
        if frame:
            if gv != pv:
                res.frame_mismatches.append(mism)
            continue
        res.compared += 1
        if gv == pv and gv is not None:
            res.agreed += 1
        else:
            res.mismatches.append(mism)


def evaluate(gt: Annotation, pred: Annotation, cfg: Config) -> FileResult:
    if (gt.width, gt.height) != (pred.width, pred.height):
        raise ValueError(f"image size differs: GT {gt.width}x{gt.height}, pred {pred.width}x{pred.height}")

    iou = iou_matrix([o.mask for o in gt.objects], [o.mask for o in pred.objects])
    pairs = dict(match(iou, cfg.min_match_iou))  # gt index -> pred index
    matched_preds = {p: g for g, p in pairs.items()}

    objects: list[ObjectResult] = []
    attrs = AttributeResult()
    for gi, g in enumerate(gt.objects):
        r = ObjectResult(g.nm, g.cls, g.area, MISSED)
        if gi in pairs:
            p = pred.objects[pairs[gi]]
            r.pred_nm, r.pred_cls, r.iou = p.nm, p.cls, float(iou[gi, pairs[gi]])
            if p.cls == g.cls:
                r.status, r.score = MATCHED, r.iou
                r.tp = r.iou >= cfg.for_class(g.cls).tp_iou
                _compare_attrs(g.attributes, p.attributes, g.nm, p.nm, attrs)
            else:
                r.status = WRONG_CLASS
        objects.append(r)
    _compare_attrs(gt.attributes, pred.attributes, None, None, attrs, frame=True)

    fps: list[FalsePositive] = []
    for pi, p in enumerate(pred.objects):
        if pi not in matched_preds:
            fps.append(FalsePositive(p.nm, p.cls, p.area, UNMATCHED))
        else:
            g = gt.objects[matched_preds[pi]]
            if g.cls != p.cls:
                fps.append(FalsePositive(p.nm, p.cls, p.area, WRONG_CLASS, g.nm, g.cls,
                                         float(iou[matched_preds[pi], pi])))

    classes: dict[str, ClassResult] = {}
    for c in sorted({o.cls for o in gt.objects} | {o.cls for o in pred.objects}):
        settings = cfg.for_class(c)
        objs = [o for o in objects if o.gt_cls == c]
        n_pred = sum(1 for o in pred.objects if o.cls == c)
        tp = sum(o.tp for o in objs)
        gt_w = sum(cfg.weigh(o.area) for o in objs)
        fp_w = sum(cfg.weigh(f.area) for f in fps if f.cls == c)
        if not objs:
            score = 0.0  # predictions only: nothing in this class was right
        else:
            score = sum(cfg.weigh(o.area) * o.score for o in objs) / (gt_w + cfg.fp_penalty * fp_w)
        classes[c] = ClassResult(
            cls=c, weight=settings.weight, tp_iou=settings.tp_iou,
            n_gt=len(objs), n_pred=n_pred, tp=tp, score=score,
            precision=tp / n_pred if n_pred else None,
            recall=tp / len(objs) if objs else None,
            gt_weight=gt_w, fp_weight=fp_w,
        )

    total_w = sum(c.weight for c in classes.values())
    if not classes:
        file_score = 1.0
    elif total_w == 0:
        raise ValueError("all classes in this frame have weight 0")
    else:
        file_score = sum(c.weight * c.score for c in classes.values()) / total_w

    return FileResult(file_score, classes, objects, fps, attrs)
