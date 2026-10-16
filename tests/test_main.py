from pathlib import Path

import tunery.render
from tunery import main as cli


def test_render_command_uses_default_output(monkeypatch) -> None:
    calls = []

    def render(
        layout: Path,
        output: Path,
        keys: list[str],
        base: Path | None = None,
        part: list[str] | None = None,
        verbose: int = 0,
    ):
        calls.append((layout, output, keys, base, part, verbose))

    monkeypatch.setattr(cli, "render", render)

    cli.main(["render", "setlists/favorites.yaml"])

    assert calls == [
        (
            Path("setlists/favorites.yaml"),
            Path("setlists/favorites.pdf"),
            [],
            None,
            None,
            0,
        )
    ]


def test_render_command_accepts_options(monkeypatch) -> None:
    calls = []

    def render(
        layout: Path,
        output: Path,
        keys: list[str],
        base: Path | None = None,
        part: list[str] | None = None,
        verbose: int = 0,
    ):
        calls.append((layout, output, keys, base, part, verbose))

    monkeypatch.setattr(cli, "render", render)

    cli.main(
        [
            "render",
            "favorites.yaml",
            "-o",
            "book.pdf",
            "-k",
            "Bb",
            "--keys",
            "Eb,F",
            "--base",
            "proj",
            "--part",
            "Piano",
            "--part",
            "Bass,Piano",
            "-v",
            "--verbose",
            "1",
        ]
    )

    assert calls == [
        (
            Path("favorites.yaml"),
            Path("book.pdf"),
            ["Bb", "Eb,F"],
            Path("proj"),
            ["Piano", "Bass,Piano"],
            1,
        )
    ]


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
