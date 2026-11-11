from pathlib import Path

import pytest

import tunery.render
from tunery import main as cli
from tunery.settings import Settings


def test_render_command_uses_default_output(monkeypatch) -> None:
    calls = []

    def render(
        layout: Path,
        output: Path,
        base: Path | None = None,
        part: list[str] | None = None,
        verbose: int = 0,
        test: list[str] | None = None,
        settings: Settings | None = None,
    ):
        calls.append(
            (
                layout,
                output,
                base,
                part,
                verbose,
                test,
                settings,
            )
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
            None,
            Settings(),
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
        test: list[str] | None = None,
        settings: Settings | None = None,
    ):
        calls.append(
            (
                layout,
                output,
                base,
                part,
                verbose,
                test,
                settings,
            )
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
            "-p",
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
            None,
            Settings(explicit_bookmark_hierarchy=True),
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


def test_render_command_forwards_test_option(monkeypatch) -> None:
    calls = []

    def render(
        layout: Path,
        output: Path,
        base: Path | None = None,
        part: list[str] | None = None,
        verbose: int = 0,
        test: list[str] | None = None,
        settings: Settings | None = None,
    ):
        calls.append(test)

    monkeypatch.setattr(cli, "render", render)

    cli.main(["render", "favorites.yaml", "--test", "A", "--test", "B"])

    assert calls == [["A", "B"]]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {}),
        (["--concert-sub"], {"concert_sub": True}),
        (["-cs"], {"concert_sub": True}),
        (["--no-concert-sub"], {"concert_sub": False}),
        (["-nocs"], {"concert_sub": False}),
        (["--current-dir"], {"current_dir": True}),
        (["--no-current-dir"], {"current_dir": False}),
        (["--explicit-bookmark-hierarchy"], {"explicit_bookmark_hierarchy": True}),
        (["--no-explicit-bookmark-hierarchy"], {"explicit_bookmark_hierarchy": False}),
        (
            ["-nocs", "--no-current-dir"],
            {"concert_sub": False, "current_dir": False},
        ),
    ],
)
def test_render_command_forwards_only_given_settings(
    monkeypatch, flags: list[str], expected: dict[str, bool]
) -> None:
    calls = []

    def render(*args, settings: Settings | None = None, **kwargs):
        assert settings is not None
        calls.append(settings.model_dump(exclude_unset=True))

    monkeypatch.setattr(cli, "render", render)

    cli.main(["render", "favorites.yaml", *flags])

    assert calls == [expected]


@pytest.mark.parametrize("flag", ["--current", "--no-current"])
def test_render_command_rejects_removed_current_option(flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.main(["render", "favorites.yaml", flag])
