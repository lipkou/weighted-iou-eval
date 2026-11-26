"""Human-readable (Russian) and JSON reports for a FileResult."""

from __future__ import annotations

import json
from dataclasses import asdict

from .metrics import MATCHED, MISSED, WRONG_CLASS, FileResult


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = lambda r: "  ".join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip()
    return [fmt(header), fmt(["-" * w for w in widths]), *map(fmt, rows)]


def _status(o) -> str:
    if o.status == MATCHED:
        return "совпал"
    if o.status == WRONG_CLASS:
        return f"нашёл, но не тот класс: {o.pred_cls}"
    return "пропущен"


def to_text(res: FileResult, title: str = "") -> str:
    out = []
    if title:
        out.append(title)
    out.append(f"ОЦЕНКА ФАЙЛА: {_pct(res.score)}")
    out.append("")

    out.append("По классам:")
    rows = []
    for c in res.classes.values():
        rows.append([c.cls, f"{c.weight:g}", _pct(c.score), _pct(c.precision), _pct(c.recall),
                     f"{c.tp_iou:g}", f"{c.tp}/{c.n_gt}/{c.n_pred}"])
    out += _table(["класс", "вес", "оценка", "precision", "recall", "порог TP", "TP/эталон/предск."], rows)
    out.append("")

    out.append("По объектам эталона:")
    rows = []
    for o in sorted(res.objects, key=lambda o: (o.gt_cls, -o.area)):
        rows.append([o.gt_nm, o.gt_cls, o.area, o.pred_nm or "—",
                     f"{o.iou:.3f}" if o.status != MISSED else "—",
                     _pct(o.score), "да" if o.tp else "нет", _status(o)])
    out += _table(["GT", "класс", "площадь", "pred", "IoU", "оценка", "TP", "статус"], rows)

    if res.false_positives:
        out.append("")
        out.append("Ложные срабатывания (FP):")
        for f in res.false_positives:
            if f.reason == WRONG_CLASS:
                why = f"неверный класс: в эталоне {f.gt_nm} — {f.gt_cls} (IoU {f.iou:.3f})"
            else:
                why = "нет объекта в эталоне"
            out.append(f"  pred {f.pred_nm}  {f.cls}  площадь {f.area}  — {why}")

    a = res.attributes
    out.append("")
    if a.compared:
        out.append(f"Атрибуты объектов: совпало {a.agreed} из {a.compared} ({_pct(a.agreement)}), в оценку не входят")
    else:
        out.append("Атрибуты объектов: сравнивать нечего")
    for m in a.mismatches:
        out.append(f"  GT {m.gt_nm} ↔ pred {m.pred_nm}: {m.key}: эталон {m.gt_value!r}, предсказание {m.pred_value!r}")
    out.append("")
    if a.frame_mismatches:
        out.append("Атрибуты кадра: РАСХОЖДЕНИЕ")
        for m in a.frame_mismatches:
            out.append(f"  {m.key}: эталон {m.gt_value!r}, предсказание {m.pred_value!r}")
    else:
        out.append("Атрибуты кадра: совпадают")
    return "\n".join(out)


def to_json(res: FileResult) -> str:
    data = asdict(res)
    data["attributes"]["agreement"] = res.attributes.agreement
    return json.dumps(data, ensure_ascii=False, indent=2)
