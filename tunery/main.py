"""Tunery CLI - Build PDF setbooks from sheet music collections."""

import argparse
from pathlib import Path
from typing import Sequence

from tunery.index import Index
from tunery.render import render


DEFAULT_INDEX_PATH = Path.home() / ".cache" / "tunery" / "index.sqlite"


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tunery: Build PDF setbooks from sheet music collections"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command: tunery index <index.json>
    index_parser = subparsers.add_parser(
        "index", help="Build index from the JSON file"
    )
    index_parser.add_argument(
        "index_json", type=Path, help="Path to the main index.json file"
    )

    # Render command: tunery render <layout.yaml> [-o output.pdf] [--index <path>]
    render_parser = subparsers.add_parser(
        "render", help="Render a PDF setbook from a YAML layout file"
    )
    render_parser.add_argument(
        "layout", type=Path, help="Path to the input YAML layout file"
    )
    render_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output PDF file (default: <input_base>.pdf)",
    )
    render_parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"Path to the index SQLite file (default: {DEFAULT_INDEX_PATH})",
    )
    render_parser.add_argument(
        "--override",
        type=Path,
        default=None,
        help="Override directory: if <title>.pdf exists here, use it instead of index lookup",
    )

    return parser.parse_args(args)


def cmd_index(args: argparse.Namespace) -> None:
    """Handle the 'index' subcommand."""
    Index.build(args.index_json, DEFAULT_INDEX_PATH)


def cmd_render(args: argparse.Namespace) -> None:
    """Handle the 'render' subcommand."""
    output_path = (
        args.output if args.output else args.layout.with_name(f"{args.layout.stem}.pdf")
    )
    render(args.layout, output_path, args.index, args.override)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)

    if parsed.command == "index":
        cmd_index(parsed)
    elif parsed.command == "render":
        cmd_render(parsed)


if __name__ == "__main__":
    main()
