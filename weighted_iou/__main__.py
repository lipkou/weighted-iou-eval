"""python -m weighted_iou GROUND_TRUTH.json PREDICTED.json [--config config.toml] [--json]"""

from __future__ import annotations

import argparse
import sys

from .annotation import load_annotation
from .config import load_config
from .metrics import evaluate
from .report import to_json, to_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="weighted_iou", description=__doc__)
    ap.add_argument("ground_truth", help="ground-truth annotation JSON")
    ap.add_argument("predicted", help="predicted annotation JSON")
    ap.add_argument("--config", default="config.toml", help="TOML settings (default: config.toml)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = evaluate(load_annotation(args.ground_truth), load_annotation(args.predicted), load_config(args.config))
    print(to_json(res) if args.json else to_text(res, f"{args.ground_truth}  vs  {args.predicted}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
