import json
from pathlib import Path

from weighted_iou.__main__ import main
from weighted_iou.metrics import evaluate
from weighted_iou.report import to_text

from test_synthetic import CFG, ann, box

ROOT = Path(__file__).resolve().parents[1]
GT = str(ROOT / "road_data" / "ground_truth" / "road.jpg.json")
PRED = str(ROOT / "road_data" / "annotated" / "road.jpg.json")
CONFIG = str(ROOT / "config.toml")


def test_cli_text(capsys):
    assert main([GT, PRED, "--config", CONFIG]) == 0
    out = capsys.readouterr().out
    assert "ОЦЕНКА ФАЙЛА: 56.0%" in out
    assert "пропущен" in out
    assert "pred 0.5  road_sign  площадь 168  — нет объекта в эталоне" in out
    assert "Атрибуты кадра: РАСХОЖДЕНИЕ" in out
    assert "weather: эталон 'clear', предсказание 'fog'" in out


def test_cli_json(capsys):
    assert main([GT, PRED, "--config", CONFIG, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert round(data["score"] * 100, 4) == 55.9912
    assert data["attributes"]["agreement"] == 1 / 3


def test_wrong_class_is_visible():
    res = evaluate(ann(("a", "car", box(0, 0, 4, 4))), ann(("x", "truck", box(0, 0, 4, 4))), CFG)
    text = to_text(res)
    assert "нашёл, но не тот класс: truck" in text
    assert "неверный класс: в эталоне a — car (IoU 1.000)" in text
