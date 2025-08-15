def multi_level_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merges two dictionaries, updating dict1 with values from dict2.

    If a key exists in both dictionaries and both values are dictionaries, it merges them recursively.
    Otherwise, it updates dict1 with the value from dict2. This means that dict2's values will
    overwrite dict1's values for the same keys.

    Args:
        dict1 (dict): The first dictionary to merge into.
        dict2 (dict): The second dictionary whose values will be merged into dict1.

    Returns:
        dict: The updated dict1 after merging with dict2.
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            multi_level_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1
