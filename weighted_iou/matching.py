"""Class-agnostic one-to-one matching of predicted objects to ground truth."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    """IoU of every GT mask (rows) with every predicted mask (columns)."""
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)))
    g = np.stack([m.ravel() for m in gt_masks]).astype(np.float64)
    p = np.stack([m.ravel() for m in pred_masks]).astype(np.float64)
    inter = g @ p.T
    union = g.sum(1)[:, None] + p.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(iou: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """Pairs (gt index, pred index) maximizing total IoU; pairs below min_iou are dropped.

    Class is deliberately ignored here: a prediction with the right shape but
    the wrong type must still be found, so it can be reported as such.
    """
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(iou, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= min_iou and iou[r, c] > 0]
