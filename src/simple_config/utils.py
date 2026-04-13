def merge_dicts(base: dict, override: dict, merge_lists = False) -> dict:
    """Recursively merge two dictionaries

    Args:
        base: The dictionary to be augmented
        override: The overriding configuration dictionary.

    Returns:
        The base dictionary with overridden keys.
    """
    for key, value in override.items():
        if ( key in base and isinstance(base[key], dict) and isinstance(value, dict) ):
            # Recursively merge dicts
            base[key] = merge_dicts(base[key], value)
        elif (
              merge_lists and
              key in base and isinstance(base[key], list) and isinstance(value, list)
             ):
            base[key] = merge_list(base[key], value)
        else:
            # Fallback to override
            base[key] = value

    return base

def merge_list(base: list, override: list) -> list:
    for value in override:
        if not base.__contains__(value):
            base.append(value)

    return base
