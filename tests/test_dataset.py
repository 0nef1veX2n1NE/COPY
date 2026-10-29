import json

import numpy as np
import pytest

from chord_estimator.config import FeatureConfig
from chord_estimator.dataset import IGNORE_INDEX, Interval, align_labels, load_manifest


def test_unsupported_reference_is_masked_not_n():
    labels, counts = align_labels([Interval(0.0, 0.2, "C:7"), Interval(0.2, 0.4, "N")], 4, FeatureConfig(sample_rate=10, hop_length=1))
    assert labels[0] == IGNORE_INDEX
    assert labels[2] == 24
    assert counts["unsupported_frames"] == 2


def test_manifest_rejects_group_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"recording_id": "a", "group_id": "same", "audio": "a.wav", "annotation": "a.lab", "split": "train"},
        {"recording_id": "b", "group_id": "same", "audio": "b.wav", "annotation": "b.lab", "split": "test"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Split leakage"):
        load_manifest(path)

