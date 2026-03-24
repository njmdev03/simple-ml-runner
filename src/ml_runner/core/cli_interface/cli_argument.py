from dataclasses import dataclass
from typing import Any, Callable, Optional, Type


@dataclass
class CLIArgument:
    name: str                      # e.g. "--lr"
    dest: Optional[str] = None     # argparse dest
    help: str = ""
    type: Optional[Type] = None
    default: Any = None
    action: Optional[str] = None   # store_true, append, etc.
    choices: Optional[list] = None
    nargs: Optional[str] = None

    # Config override behavior
    config_path: Optional[str] = None

    # Optional transformation (for typer or special cases)
    callback: Optional[Callable] = None

    # For boolean inversion flags like --dont-train
    invert: bool = False
