"""Tunery CLI - Build PDF setbooks from sheet music collections."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

import tunery.render as render_module
from tunery.render import render

app = App(help="Tunery: Build PDF setbooks from sheet music collections")


@app.command(name="render")
def render_command(
    layout: Annotated[Path, Parameter(help="Path to the input YAML layout file")],
    output: Annotated[
        Path | None,
        Parameter(
            ["--output", "-o"],
            help="Path to the output PDF file (default: <input_base>.pdf)",
        ),
    ] = None,
    base: Annotated[
        Path | None,
        Parameter(
            ["--base"],
            help="Base directory: relative paths on the command line resolve against it (default: working directory).",
        ),
    ] = None,
    part: Annotated[
        list[str] | None,
        Parameter(
            ["--part"],
            help="Requested part(s) with substitution chains. Repeat for multiple outputs; comma combines in one file.",
        ),
    ] = None,
    verbose: Annotated[
        int,
        Parameter(
            ["--verbose"],
            help="Verbosity level 0-2 (default 0).",
        ),
    ] = 0,
    v: Annotated[
        int,
        Parameter(
            ["-v"],
            count=True,
            help="Increase verbosity (-v level 1, -vv level 2).",
        ),
    ] = 0,
    explicit_bookmark_hierarchy: Annotated[
        bool,
        Parameter(
            ["--explicit-bookmark-hierarchy"],
            negative=(),
            help='Prefix part bookmarks with "_ " for readers that show one bookmark level.',
        ),
    ] = False,
    test: Annotated[
        list[str] | None,
        Parameter(
            ["--test"],
            help="Resolve only these titles and print -vv results without writing a PDF. Repeatable; '.' or '' means every title.",
        ),
    ] = None,
) -> None:
    """Render a PDF setbook from a YAML layout file."""
    output_path = output if output else layout.with_name(f"{layout.stem}.pdf")
    level = max(verbose, v)
    render(
        layout,
        output_path,
        base,
        part or None,
        level,
        explicit_bookmark_hierarchy=explicit_bookmark_hierarchy,
        test=test,
    )


@app.command(name="lookup")
def lookup_command(
    title: Annotated[
        list[str],
        Parameter(
            help="One or more tune titles to search for",
            negative_iterable=(),
        ),
    ],
    output: Annotated[
        Path | None,
        Parameter(
            ["--output", "-o"],
            help="Output path: if file, use as-is; if directory, save TITLE.pdf there; default: TITLE.pdf in cwd",
        ),
    ] = None,
    base: Annotated[
        Path | None,
        Parameter(
            ["--base"],
            help="Base directory: configuration is found from it and relative paths resolve against it (default: working directory).",
        ),
    ] = None,
) -> None:
    """Look up titles in configured libraries and extract them to PDF."""

    for item in title:
        render_module.lookup_and_extract(item, output, base)


def main(args: Sequence[str] | None = None) -> None:
    app(args, result_action="return_value")


if __name__ == "__main__":
    main()
