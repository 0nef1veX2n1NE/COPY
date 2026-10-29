import json
import numpy as np

from chord_estimator.config import FeatureConfig
from chord_estimator.decoding import decode_segments, write_outputs
from chord_estimator.labels import LABELS_25


def test_low_confidence_becomes_unknown_not_n(tmp_path):
    logits = np.zeros((3, 25), np.float32)
    segments, probabilities = decode_segments(logits, LABELS_25, FeatureConfig(sample_rate=10, hop_length=1), 0.3, review_threshold=0.5)
    assert len(segments) == 1
    assert segments[0].chord == "UNKNOWN"
    assert segments[0].needs_review
    paths = write_outputs(tmp_path / "result", segments, logits, probabilities, {"test": True}, LABELS_25)
    document = json.loads(open(paths["json"], encoding="utf-8").read())
    assert "uncalibrated" in document["score_semantics"]


def test_adjacent_equal_frames_merge_and_top_candidates_are_kept():
    logits = np.full((4, 25), -5.0, np.float32)
    logits[:2, 0] = 5
    logits[2:, 3] = 5
    segments, _ = decode_segments(logits, LABELS_25, FeatureConfig(sample_rate=10, hop_length=1), 0.4, top_k=3)
    assert [segment.chord for segment in segments] == [LABELS_25[0], LABELS_25[3]]
    assert all(len(segment.candidates) == 3 for segment in segments)

