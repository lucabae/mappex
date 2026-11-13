"""Tests for dataset assembly: annotation conversion and the train/val split."""

import json
from pathlib import Path

from mappex.train import _annotations, split


def test_annotations_become_yolo_lines():
    sample = {
        "labels": [
            {"class": "chair", "bbox_xywh_normalized": [0.1, 0.2, 0.2, 0.4]},
        ]
    }
    lines = _annotations(sample, {"chair": 5})
    assert lines == ["5 0.200000 0.400000 0.200000 0.400000"]


def test_unknown_classes_are_skipped():
    sample = {"labels": [{"class": "spaceship", "bbox_xywh_normalized": [0, 0, 1, 1]}]}
    assert _annotations(sample, {"chair": 0}) == []


def test_boxes_are_clamped_inside_the_image():
    sample = {"labels": [{"class": "chair", "bbox_xywh_normalized": [0.9, 0.9, 0.5, 0.5]}]}
    (line,) = _annotations(sample, {"chair": 0})
    _, cx, cy, width, height = (float(v) for v in line.split())
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert 0.0 < width <= 1.0 and 0.0 < height <= 1.0


def test_split_keeps_every_class_in_both_splits():
    per_class = {
        "chair": [{"id": i} for i in range(10)],
        "bench": [{"id": i} for i in range(10)],
    }
    train, val = split(per_class, val_ratio=0.2, seed=1)
    assert len(train) == 16
    assert len(val) == 4


def test_split_handles_a_single_sample():
    train, val = split({"chair": [{"id": 0}]})
    assert len(train) == 1
    assert val == []


def test_split_is_deterministic():
    per_class = {"chair": [{"id": i} for i in range(20)]}
    first = split(per_class, seed=7)
    second = split({"chair": [{"id": i} for i in range(20)]}, seed=7)
    assert [s["id"] for s in first[1]] == [s["id"] for s in second[1]]
