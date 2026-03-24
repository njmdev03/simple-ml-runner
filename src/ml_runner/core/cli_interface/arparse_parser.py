import argparse

from ml_runner.core.utils.dict_utils import set_in_dict
from ml_runner.core.cli_interface.base_parser import BaseParser
from ml_runner.core.cli_interface.cli_argument import CLIArgument


class ArgparseParser(BaseParser):

    def __init__(self, registry=None):
        super().__init__()
        self.parser = argparse.ArgumentParser(description="ML Runner CLI")
        self.args = None
        if registry:
            self.add_arguments(registry.get_arguments())

    def add_argument(self, arg: CLIArgument):
        super().add_argument(arg)
        kwargs = {
            "help": arg.help,
            "default": argparse.SUPPRESS
        }

        if arg.type:
            kwargs["type"] = arg.type
        if arg.action:
            kwargs["action"] = arg.action
        if arg.choices:
            kwargs["choices"] = arg.choices
        if arg.dest:
            kwargs["dest"] = arg.dest
        if getattr(arg, 'nargs', None) is not None:
            kwargs['nargs'] = arg.nargs

        self.parser.add_argument(arg.name, **kwargs)

    def parse_args(self) -> dict:
        self.args = self.parser.parse_args()
        return vars(self.args)

    def get_config_overrides(self):
        overrides = {}

        for arg in self.registered_arguments:
            if not arg.config_path:
                continue

            attr = arg.dest or arg.name.lstrip('-').replace('-', '_')

            # Since we use argparse.SUPPRESS, the attribute will only exist
            # if the user provided it on the command line.
            if hasattr(self.args, attr):
                value = getattr(self.args, attr)
                if arg.invert:
                    value = not value
                set_in_dict(overrides, arg.config_path, value)

        return overrides
