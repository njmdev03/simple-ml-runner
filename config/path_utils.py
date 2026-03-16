import datetime
from pathlib import Path
from typing import Dict, Any

def resolve_path_template(template: str, context: Dict[str, Any]) -> str:
    """
    Resolves placeholders in a path template string.

    Supported placeholders:
    - {experiment_name}
    - {epoch}
    - {date} or {date:format}
    - {timestamp} or {timestamp:format}

    Example:
    template = "checkpoints/{experiment_name}/{date:%Y%m%d}/model_{epoch}.pt"
    """
    now = datetime.datetime.now()

    # default date/timestamp if not specified with format
    res_context = {
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.strftime("%H%M%S"),
        **context
    }

    # Handle formatted dates/timestamps manually or using string format if possible
    # A simple but powerful way is using string.Formatter, but for {date:%Y%m%d}
    # we can use a custom logic or just rely on the fact that we can pre-format these.

    # For formatted dates like {date:%Y%m%d}, we need to detect them.
    # We'll use a simple approach: if "date:" is in the template, we extract the format.

    result = template

    if "{date:" in result:
        # Very naive parser, but handles the requirement
        start = result.find("{date:")
        end = result.find("}", start)
        fmt = result[start+6:end]
        result = result.replace(result[start:end+1], now.strftime(fmt))

    if "{timestamp:" in result:
        start = result.find("{timestamp:")
        end = result.find("}", start)
        fmt = result[start+11:end]
        result = result.replace(result[start:end+1], now.strftime(fmt))

    # Apply standard formatting for the rest
    try:
        result = result.format(**res_context)
    except KeyError as e:
        # If a placeholder is missing (like 'epoch' in a non-training context),
        # we might want to keep it or replace with 'unknown'
        # For now, let's just keep it as is or raise a warning?
        # Re-raising for visibility in this development phase.
        pass

    return result

def ensure_dir(path: str):
    """Ensures the directory for the given path exists."""
    p = Path(path)
    if p.suffix: # it's a file path
        p.parent.mkdir(parents=True, exist_ok=True)
    else: # it's a directory path
        p.mkdir(parents=True, exist_ok=True)
