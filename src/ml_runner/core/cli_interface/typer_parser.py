import typer
import click
import sys

from ml_runner.core.utils.dict_utils import set_in_dict
from ml_runner.core.cli_interface.base_parser import BaseParser
from ml_runner.core.cli_interface.cli_argument import CLIArgument


class TyperParser(BaseParser):

    def __init__(self, registry=None):
        super().__init__()
        self.app = typer.Typer()
        self.args_dict = {}
        self.cfg_dict = {}

        if registry:
            self.add_arguments(registry.get_arguments())

    def add_argument(self, arg: CLIArgument):
        super().add_argument(arg)

    def parse_args(self):
        def main_callback(**kwargs):
            ctx = click.get_current_context()
            self.args_dict = kwargs
            # Capture the source of each parameter to identify what was explicitly set
            self.param_sources = {
                name: ctx.get_parameter_source(name)
                for name in kwargs
            }

        # Create a Click command manually
        click_command = click.Command(
            name='ml-runner',
            callback=main_callback,
            help="ML Runner CLI"
        )

        for arg in self.registered_arguments:
            param_decls = [arg.name]
            if arg.dest:
                # Click uses the last non-dashed name as the internal name
                param_decls.append(arg.dest)

            kwargs = {
                "help": arg.help,
                # Set default but we will filter it out in get_config_overrides if it wasn't explicitly provided
                "default": arg.default if arg.default is not None else (False if arg.action == "store_true" else None),
                "type": arg.type,
                "multiple": arg.action == "append",
                "is_flag": arg.action in ["store_true", "store_false"],
            }

            # Click uses different names for some parameters
            if arg.action == "store_true":
                kwargs["flag_value"] = True
            elif arg.action == "store_false":
                kwargs["flag_value"] = False

            # Remove None values to let Click handle defaults
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            option = click.Option(param_decls, **kwargs)
            click_command.params.append(option)

        # Run the click command directly
        try:
            click_command.main(args=sys.argv[1:], standalone_mode=False)
        except SystemExit:
            pass

        return self.args_dict

    def get_config_overrides(self):
        import click
        overrides = {}
        for arg in self.registered_arguments:
            if not arg.config_path:
                continue

            attr = arg.dest or arg.name.lstrip('-').replace('-', '_')
            value = self.args_dict.get(attr)

            # Check if the parameter was explicitly provided on the command line
            source = getattr(self, 'param_sources', {}).get(attr)
            if source == click.core.ParameterSource.COMMANDLINE:
                val = not value if arg.invert else value
                set_in_dict(overrides, arg.config_path, val)

        return overrides
