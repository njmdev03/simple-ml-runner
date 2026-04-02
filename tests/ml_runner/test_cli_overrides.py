import pytest
import sys
from unittest.mock import patch
from ml_runner.core.cli_interface.cli_argument import CLIArgument
from ml_runner.core.cli_interface.cli_registry import CLIRegistry
from ml_runner.core.cli_interface.arparse_parser import ArgparseParser
from ml_runner.core.cli_interface.typer_parser import TyperParser


def test_argparse_overrides():
    CLIRegistry.clear()
    CLIRegistry.register(CLIArgument("--lr", type=float, help="LR", config_path="opt.lr"))
    CLIRegistry.register(CLIArgument("--train", action="store_true", dest="do_train", config_path="train.enabled"))

    # CASE: No arguments passed
    with patch.object(sys, 'argv', ['prog']):
        parser = ArgparseParser(CLIRegistry)
        parser.parse_args()
        overrides = parser.get_config_overrides()
        assert overrides == {}

    # CASE: Single override passed
    with patch.object(sys, 'argv', ['prog', '--lr', '0.01']):
        parser = ArgparseParser(CLIRegistry)
        parser.parse_args()
        overrides = parser.get_config_overrides()
        assert overrides == {'opt': {'lr': 0.01}}

    # CASE: Flag passed
    with patch.object(sys, 'argv', ['prog', '--train']):
        parser = ArgparseParser(CLIRegistry)
        parser.parse_args()
        overrides = parser.get_config_overrides()
        assert overrides == {'train': {'enabled': True}}


def test_typer_overrides():
    # Only test if typer is available
    try:
        import typer
        import click
    except ImportError:
        pytest.skip("Typer or Click not installed")

    CLIRegistry.clear()
    CLIRegistry.register(CLIArgument("--lr", type=float, help="LR", config_path="opt.lr"))
    CLIRegistry.register(CLIArgument("--train", action="store_true", dest="do_train", config_path="train.enabled"))
    CLIRegistry.register(CLIArgument("--dont-train", action="store_false", dest="do_train", config_path="train.enabled"))

    # CASE: No arguments passed
    # Typer/Click might be tricky with sys.argv patch because of how they handle context
    # But let's try.
    with patch.object(sys, 'argv', ['prog']):
        parser = TyperParser(CLIRegistry)
        parser.parse_args()
        overrides = parser.get_config_overrides()
        # Should be empty because no command line args were provided
        assert overrides == {}

    # CASE: Single override passed
    with patch.object(sys, 'argv', ['prog', '--lr', '0.01']):
        parser.parse_args()
        overrides = parser.get_config_overrides()
        assert overrides == {'opt': {'lr': 0.01}}

    # CASE: Flag passed
    with patch.object(sys, 'argv', ['prog', '--train']):
        parser = TyperParser(CLIRegistry)
        parser.parse_args()
        overrides = parser.get_config_overrides()
        assert overrides == {'train': {'enabled': True}}

    # CASE: Inverse flag passed
    with patch.object(sys, 'argv', ['prog', '--dont-train']):
        parser = TyperParser(CLIRegistry)
        parser.parse_args()
        overrides = parser.get_config_overrides()
        assert overrides == {'train': {'enabled': False}}
