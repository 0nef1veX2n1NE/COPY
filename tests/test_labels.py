from chord_estimator.labels import LABELS_25, map_to_majmin, transpose_internal


def test_btc_compatible_label_order():
    assert len(LABELS_25) == 25
    assert LABELS_25[:4] == ("C:maj", "C:min", "C#:maj", "C#:min")
    assert LABELS_25[-1] == "N"


def test_n_and_unknown_are_distinct():
    assert map_to_majmin("N").internal == "N"
    unknown = map_to_majmin("UNKNOWN")
    assert not unknown.supported and unknown.internal is None


def test_unsupported_quality_is_not_n():
    result = map_to_majmin("C:7")
    assert not result.supported
    assert result.internal is None


def test_transposition_wraps_and_preserves_quality():
    assert transpose_internal("B:min", 2) == "C#:min"
    assert transpose_internal("N", -5) == "N"

