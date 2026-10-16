from pathlib import Path

import pytest

import tunery.render
from tunery import main as cli


def test_render_command_uses_default_output(monkeypatch) -> None:
    calls = []

    def render(
        layout: Path,
        output: Path,
        base: Path | None = None,
        part: list[str] | None = None,
        verbose: int = 0,
        explicit_bookmark_hierarchy: bool = False,
    ):
        calls.append(
            (layout, output, base, part, verbose, explicit_bookmark_hierarchy)
        )

    monkeypatch.setattr(cli, "render", render)

    cli.main(["render", "setlists/favorites.yaml"])

    assert calls == [
        (
            Path("setlists/favorites.yaml"),
            Path("setlists/favorites.pdf"),
            None,
            None,
            0,
            False,
        )
    ]


def test_render_command_accepts_options(monkeypatch) -> None:
    calls = []

    def render(
        layout: Path,
        output: Path,
        base: Path | None = None,
        part: list[str] | None = None,
        verbose: int = 0,
        explicit_bookmark_hierarchy: bool = False,
    ):
        calls.append(
            (layout, output, base, part, verbose, explicit_bookmark_hierarchy)
        )

    monkeypatch.setattr(cli, "render", render)

    cli.main(
        [
            "render",
            "favorites.yaml",
            "-o",
            "book.pdf",
            "--base",
            "proj",
            "--part",
            "Piano",
            "--part",
            "Bass,Piano",
            "-v",
            "--verbose",
            "1",
            "--explicit-bookmark-hierarchy",
        ]
    )

    assert calls == [
        (
            Path("favorites.yaml"),
            Path("book.pdf"),
            Path("proj"),
            ["Piano", "Bass,Piano"],
            1,
            True,
        )
    ]


def test_render_command_rejects_removed_keys_option() -> None:
    with pytest.raises(SystemExit):
        cli.main(["render", "favorites.yaml", "-k", "Bb"])


def test_lookup_command_accepts_options(monkeypatch) -> None:
    calls = []

    def lookup_and_extract(title: str, output: Path | None, base: Path | None = None):
        calls.append((title, output, base))

    monkeypatch.setattr(tunery.render, "lookup_and_extract", lookup_and_extract)

    cli.main(["lookup", "Blue Monk", "-o", "blue.pdf", "--base", "proj"])

    assert calls == [("Blue Monk", Path("blue.pdf"), Path("proj"))]


def test_lookup_command_accepts_multiple_titles(monkeypatch) -> None:
    calls = []

    def lookup_and_extract(title: str, output: Path | None, base: Path | None = None):
        calls.append((title, output, base))

    monkeypatch.setattr(tunery.render, "lookup_and_extract", lookup_and_extract)

    cli.main(["lookup", "Blue Monk", "Autumn Leaves"])

    assert calls == [
        ("Blue Monk", None, None),
        ("Autumn Leaves", None, None),
    ]
