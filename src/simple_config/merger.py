from copy import deepcopy

def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries according to simple_config rules.

    Rules:
    - Dict + Dict -> Recursive merge.
    - Lists -> REPLACE (never merge lists).
    - Scalars -> Override.

    Args:
        base: The base configuration dictionary.
        override: The overriding configuration dictionary.

    Returns:
        A new dictionary with the merged data. The input dictionaries are not modified.
    """
    # Create a deep copy of the base to avoid mutating input data
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge dicts
            result[key] = merge_dicts(result[key], value)
        else:
            # For everything else (scalars, lists, etc.), override with the new value
            # Note: We deepcopy the override value too if it's a dict/list to ensure 
            # the result is fully independent.
            result[key] = deepcopy(value)

    return result
