"""CLI interface for ritm_annotation project."""

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from ritm_annotation.utils.misc import load_module
from gettext import gettext as _
import gettext

import faulthandler

faulthandler.enable()

logger = logging.getLogger(__name__)

locale_dir = Path(__file__).parent.parent / "i18n"

gettext.bindtextdomain(
    "ritm_annotation",
    localedir=str(locale_dir),
)

logger.debug(
    _('Loading locale data from "{locale_folder}"').format(locale_folder=locale_dir)
)


def add_subcommand(subparsers, name: str, submodule):
    subparser = subparsers.add_parser(
        name,
        help=submodule.COMMAND_DESCRIPTION,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    common_flags(subparser)
    handler = submodule.command(subparser)
    subparser.set_defaults(fn=handler)


def common_flags(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help=_("Give more details about what is happening"),
    )  # noqa: E501
    parser.add_argument(
        "-V",
        "--version",
        dest="is_show_version",
        action="store_true",
        help=_("Print version and exit"),
    )  # noqa: E501


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m ritm_annotation` and `$ ritm_annotation `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    logging.basicConfig()
    parser = ArgumentParser(
        prog="ritm_annotation", formatter_class=ArgumentDefaultsHelpFormatter
    )
    common_flags(parser)
    subparsers = parser.add_subparsers()

    for module in Path(__file__).parent.glob("*/__init__.py"):
        if str(module).find("pycache") > 0:
            continue
        module_name = module.parent.name
        subcommand_module = load_module(
            module, module_name=f"ritm_annotation.cli.{module_name}"
        )
        add_subcommand(subparsers, module_name, subcommand_module)

    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    version = open(str(Path(__file__).parent.parent / "VERSION"), "r").read()
    if args.is_show_version:
        print(version)
        exit(0)
    logger.debug(f"{_('Starting')} ritm_annotation v{version}")

    fn = args.__dict__.get("fn")
    args.__dict__["fn"] = None
    if fn is not None:
        fn(args)
    else:
        parser.parse_args([*sys.argv[1:], "--help"])
