"""CLI interface for ritm_annotation project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from ritm_annotation.cli.annotate import command as command_annotate

logger = logging.getLogger(__name__)


def add_subcommand(subparsers, name: str, command_fn):
    subparser = subparsers.add_parser(name)
    subparser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true"
    )
    handler = command_fn(subparser)
    subparser.set_defaults(fn=handler)


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
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    subparsers = parser.add_subparsers()
    add_subcommand(subparsers, "annotate", command_annotate)
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)

    # logging.warn("info")
    logger.debug("verbose")
    fn = args.__dict__.get("fn")
    if fn is not None:
        fn(args)
    else:
        parser.parse_args([*sys.argv[1:], "--help"])
