"""Tests for workflow methods."""

from geb.workflows.methods import multi_level_merge


def test_multi_level_merge() -> None:
    """Test multi-level dictionary merging.

    Verifies that the multi_level_merge function correctly merges
    two dictionaries, including nested dictionaries, without mutating originals.
    """
    dict1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict2 = {"a": 2, "c": {"d": 4, "f": 5}}

    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 2, "b": 2, "c": {"d": 4, "e": 4, "f": 5}}
    assert dict1 == {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    assert dict2 == {"a": 2, "c": {"d": 4, "f": 5}}


def test_multi_level_merge_deep_nesting() -> None:
    """Test merging deeply nested dictionaries."""
    dict1 = {"a": {"b": {"c": {"d": 1}}}}
    dict2 = {"a": {"b": {"c": {"e": 2}}, "f": 3}}
    merged = multi_level_merge(dict1, dict2)
    expected = {"a": {"b": {"c": {"d": 1, "e": 2}}, "f": 3}}
    assert merged == expected


def test_multi_level_merge_non_dict_values() -> None:
    """Test merging when keys have non-dictionary values."""
    dict1 = {"a": 1, "b": [1, 2], "c": "string"}
    dict2 = {"a": 2, "b": [3, 4], "d": 5}
    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 2, "b": [3, 4], "c": "string", "d": 5}


def test_multi_level_merge_mixed_types() -> None:
    """Test merging with mixed types, including overwriting dict with non-dict."""
    dict1 = {"a": {"nested": 1}}
    dict2 = {"a": "not_a_dict"}
    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": "not_a_dict"}
