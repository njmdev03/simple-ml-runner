def merge_dicts(base, override):
    result = base.copy()

    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v

    return result
