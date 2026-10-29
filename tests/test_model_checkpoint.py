from dataclasses import replace

import pytest
import torch

from chord_estimator.checkpoint import CheckpointError, load_custom_checkpoint, save_checkpoint
from chord_estimator.config import FeatureConfig, ModelConfig
from chord_estimator.models import ChordTransformer, LegacyBTCModel


def tiny_config(**changes):
    base = ModelConfig(input_bins=12, d_model=16, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.0)
    return replace(base, **changes)


def test_transformer_shapes_and_boundary_head():
    model = ChordTransformer(tiny_config(boundary_head=True))
    output = model(torch.randn(2, 7, 12), torch.ones(2, 7, dtype=torch.bool))
    assert output["chord_logits"].shape == (2, 7, 25)
    assert output["boundary_logits"].shape == (2, 7)


def test_cnn_variant_shape():
    model = ChordTransformer(tiny_config(cnn_frontend=True, architecture="cnn_transformer"))
    assert model(torch.randn(1, 9, 12))["chord_logits"].shape == (1, 9, 25)


def test_checkpoint_roundtrip_and_label_metadata(tmp_path):
    config = tiny_config()
    model = ChordTransformer(config)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, FeatureConfig(n_bins=12), {"mean": 0.5, "std": 2.0}, 3, "split-hash", {"kind": "test"})
    loaded, payload = load_custom_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, ChordTransformer)
    assert payload["split_version"] == "split-hash"


def test_missing_checkpoint_stops(tmp_path):
    with pytest.raises(CheckpointError, match="Inference is disabled"):
        load_custom_checkpoint(tmp_path / "missing.pt", torch.device("cpu"))


def test_legacy_model_rejects_non_108_input():
    model = LegacyBTCModel()
    with pytest.raises(ValueError, match="108"):
        model(torch.randn(1, 100, 144))

