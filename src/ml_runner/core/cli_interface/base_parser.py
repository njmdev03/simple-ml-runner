from typing import Any, Dict, List

from ml_runner.core.cli_interface.cli_argument import CLIArgument


class BaseParser:
    """
    Base interface for argument parsers.
    Implementations must provide:
      - parse_args(): returns dict of CLI arguments
      - get_config_overrides(): returns nested dict of overrides
      - add_argument(arg): registers a single CLIArgument
    """
    def __init__(self):
        self.registered_arguments: List[CLIArgument] = []

    def add_argument(self, arg: CLIArgument):
        """Registers a single CLIArgument."""
        self.registered_arguments.append(arg)

    def add_arguments(self, args: List[CLIArgument]):
        """Registers a list of CLIArguments."""
        for arg in args:
            self.add_argument(arg)

    def parse_args(self) -> Dict[str, Any]:
        """Returns a dictionary of CLI arguments."""
        raise NotImplementedError

    def get_config_overrides(self) -> Dict:
        raise NotImplementedError
