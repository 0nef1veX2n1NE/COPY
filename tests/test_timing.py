import numpy as np

from chord_estimator.config import FeatureConfig, btc_ismir19_feature_config
from chord_estimator.timing import frame_edges, frame_sample_indices, frame_times


def test_frame_coordinates_follow_samples_hop_and_stride():
    config = FeatureConfig(sample_rate=100, hop_length=10, time_stride=2, frame_origin_samples=5)
    np.testing.assert_array_equal(frame_sample_indices(3, config), [5, 25, 45])
    np.testing.assert_allclose(frame_times(3, config, 1.0), [1.05, 1.25, 1.45])


def test_final_edge_is_clamped_to_decoded_duration():
    config = FeatureConfig(sample_rate=100, hop_length=10)
    np.testing.assert_allclose(frame_edges(4, config, 0.35), [0.0, 0.1, 0.2, 0.3, 0.35])


def test_btc_block_boundary_uses_source_sample_index_without_drift():
    config = btc_ismir19_feature_config()
    indices = frame_sample_indices(110, config)
    assert indices[107] == 107 * 2048
    assert indices[108] == 220500
    assert indices[109] == 220500 + 2048
