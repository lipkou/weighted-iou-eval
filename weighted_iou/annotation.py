"""Loading annotation files and turning every object into a binary mask.

File format (one JSON file per image) is a list of two blocks:
  [0] frame metadata: width, height, frame attributes and the object list
      (nm, type, shape);
  [1] geometry: per object x1/y1/x2/y2, optional ``path`` (polygon rings)
      and optional object attributes.
Both blocks are joined by ``nm``. ``nm`` is only unique inside one file:
ground truth and prediction number their objects independently.

Geometry source, in order of priority:
  1. a mask PNG in ``<file>.images/<frame:05d>/<nm>.png`` (any value > 0
     is foreground; the tool writes palette PNGs with values 0/1);
  2. ``path``: polygon rings, filled and united;
  3. the bounding box x1, y1, x2, y2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize_class(name: str) -> str:
    """'road sign', 'Road-Sign' and 'road_sign' are the same class."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


@dataclass
class AnnotatedObject:
    nm: str
    cls: str
    shape: str
    mask: np.ndarray  # bool, H x W
    attributes: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Annotation:
    file: str
    width: int
    height: int
    attributes: dict
    objects: list[AnnotatedObject]


def rasterize(geom: dict, width: int, height: int) -> np.ndarray:
    """Polygon rings (united) or, without a path, the bounding box.

    Uses PIL's fill convention: polygon edges and the x2/y2 edge of the box
    are included. Ground-truth masks come from the annotation tool and may
    round differently; on objects of ~100 px this alone moves IoU by a few
    points.
    """
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    rings = geom.get("path")
    if rings:
        for ring in rings:
            if len(ring) >= 3:
                draw.polygon([tuple(p) for p in ring], fill=1)
    else:
        draw.rectangle([geom["x1"], geom["y1"], geom["x2"], geom["y2"]], fill=1)
    return np.array(img, dtype=bool)


def _mask_dir(json_path: Path, frame: int) -> Path:
    # road.jpg.json -> road.jpg.images/00001
    return json_path.with_name(json_path.name[: -len(".json")] + ".images") / f"{frame:05d}"


def load_annotation(path: str | Path) -> Annotation:
    path = Path(path)
    blocks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blocks, list) or len(blocks) < 2:
        raise ValueError(f"{path}: expected a list of [metadata, geometry] blocks")
    meta, geometry = blocks[0], blocks[1]
    width, height = int(meta["width"]), int(meta["height"])
    types = {o["nm"]: o for o in meta.get("objects", [])}
    mask_dir = _mask_dir(path, int(meta.get("original_frame", 1)))

    objects = []
    for geom in geometry.get("objects", []):
        nm = geom["nm"]
        if nm not in types:
            raise ValueError(f"{path}: object {nm} has geometry but no type in block 0")
        mask_file = mask_dir / f"{nm}.png"
        if mask_file.is_file():
            mask = np.array(Image.open(mask_file)) > 0
            if mask.ndim == 3:  # RGB(A) mask: any non-zero channel
                mask = mask.any(axis=2)
            if mask.shape != (height, width):
                raise ValueError(f"{mask_file}: size {mask.shape[::-1]} != {width}x{height}")
        else:
            mask = rasterize(geom, width, height)
        objects.append(
            AnnotatedObject(
                nm=nm,
                cls=normalize_class(types[nm]["type"]),
                shape=types[nm].get("shape", ""),
                mask=mask,
                attributes=geom.get("attributes") or {},
            )
        )

    missing = set(types) - {o.nm for o in objects}
    if missing:
        raise ValueError(f"{path}: objects without geometry: {sorted(missing)}")

    return Annotation(
        file=meta.get("file", path.name),
        width=width,
        height=height,
        attributes=meta.get("attributes") or {},
        objects=objects,
    )
