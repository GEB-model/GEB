from geb.workflows.methods import multi_level_merge


def test_multi_level_merge() -> None:
    """Test whether two dictionaries with nested structures can be merged correctly."""
    dict1 = {"a": 1, "b": {"c": 2, "d": 3}}
    dict2 = {"b": {"d": 4, "e": 5}, "f": 6}

    merged_dict = multi_level_merge(dict1, dict2)

    expected_dict = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}

    assert merged_dict == expected_dict
