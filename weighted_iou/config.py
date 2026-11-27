"""Evaluation settings, read from a TOML file (see config.toml)."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .annotation import normalize_class

OBJECT_WEIGHTS = {
    "area": lambda a: float(a),
    "sqrt_area": lambda a: math.sqrt(a),
}


@dataclass
class ClassSettings:
    weight: float
    tp_iou: float


@dataclass
class Config:
    min_match_iou: float = 0.1
    object_weight: str = "area"
    fp_penalty: float = 1.0
    default_tp_iou: float = 0.5
    # None: a class missing from [classes] is an error, not a silent default.
    default_class_weight: float | None = None
    classes: dict[str, ClassSettings] = field(default_factory=dict)

    def __post_init__(self):
        if self.object_weight not in OBJECT_WEIGHTS:
            raise ValueError(f"object_weight must be one of {sorted(OBJECT_WEIGHTS)}")
        if self.fp_penalty < 0:
            raise ValueError("fp_penalty must be >= 0")

    def weigh(self, area: int) -> float:
        return OBJECT_WEIGHTS[self.object_weight](area)

    def for_class(self, cls: str) -> ClassSettings:
        if cls in self.classes:
            return self.classes[cls]
        if self.default_class_weight is None:
            raise KeyError(
                f"class '{cls}' is not in [classes] of the config "
                "(add it or set scoring.default_class_weight)"
            )
        return ClassSettings(self.default_class_weight, self.default_tp_iou)


def load_config(path: str | Path) -> Config:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    matching = data.get("matching", {})
    scoring = data.get("scoring", {})
    default_tp = float(scoring.get("default_tp_iou", 0.5))
    classes = {}
    for name, spec in data.get("classes", {}).items():
        key = normalize_class(name)
        if key in classes:
            raise ValueError(f"class '{name}' is listed twice after normalization ('{key}')")
        classes[key] = ClassSettings(
            weight=float(spec["weight"]),
            tp_iou=float(spec.get("tp_iou", default_tp)),
        )
    dcw = scoring.get("default_class_weight")
    return Config(
        min_match_iou=float(matching.get("min_iou", 0.1)),
        object_weight=scoring.get("object_weight", "area"),
        fp_penalty=float(scoring.get("fp_penalty", 1.0)),
        default_tp_iou=default_tp,
        default_class_weight=None if dcw is None else float(dcw),
        classes=classes,
    )
