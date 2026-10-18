"""File-system view: paths as the user navigates them, through links.

A `ViewPath` is a `Path` navigated lexically: `.`/`..` collapse and
directory symlinks are never resolved. A directory symlink is a link: going
down through it enters the linked tree, and going up from inside it returns
through the link, never to the linked tree's physical parents.

Everything else is `Path`: derived paths (`/`, `parent`, `parents`,
`iterdir`, `glob`, `walk`) stay `ViewPath` and carry the same base,
`relative_to` gives message paths, and I/O goes through the links.

The base is the root of the view (`--base`): `ViewPath("gig.yaml",
base=args.base).absolute()` joins it, an absolute path ignores it, and
without a base the logical working directory is used. A relative path does
its I/O against its base, not the process directory, while `str()` shows it
as given. Equality compares the path as written, not the base.

Depends on nothing in tunery, so it can move to its own package.
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path, PurePath
from typing import Self


def _logical_cwd() -> str:
    """`$PWD` while it is an absolute path naming the current directory (a
    `cd` through a link is kept), else `os.getcwd()`."""
    pwd = os.environ.get("PWD")
    if pwd and os.path.isabs(pwd):
        pwd = os.path.normpath(pwd)
        try:
            if os.path.samefile(pwd, "."):
                return pwd
        except OSError:
            pass
    return os.getcwd()


def _text(segment: str | os.PathLike[str]) -> str:
    """A segment as written: a `ViewPath` is not made absolute."""
    return str(segment) if isinstance(segment, PurePath) else os.fspath(segment)


def _identity(path: Path) -> tuple[int, int] | None:
    """Which real directory `path` is, or None when it can't be told."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino)


class ViewPath(Path):
    _base: "ViewPath | None"

    def __init__(
        self,
        *args: str | os.PathLike[str],
        base: str | os.PathLike[str] | None = None,
    ) -> None:
        """Join `args` and collapse `.`/`..` lexically; derived paths go
        through here too, so `p / ".."` is `p.parent` even inside a link.
        `base` is made absolute against the logical working directory."""
        joined = os.path.normpath(os.path.join(*map(_text, args))) if args else "."
        super().__init__(joined)
        if base is None:
            self._base = None
        elif isinstance(base, ViewPath) and base._base is base:
            self._base = base
        else:
            rooted = ViewPath(base).absolute()
            rooted._base = rooted
            self._base = rooted

    @property
    def base(self) -> Self:
        """Root of the view: the given base, else the logical working
        directory. It is absolute, and its own base."""
        if self._base is not None:
            return self._base  # type: ignore[return-value]
        rooted = type(self)(_logical_cwd())
        rooted._base = rooted
        return rooted

    def with_segments(self, *pathsegments: str | os.PathLike[str]) -> Self:
        """Derived paths keep this path's base (the `Path` hook for this)."""
        return type(self)(*pathsegments, base=self._base)

    def __fspath__(self) -> str:
        """The absolute path, so I/O never depends on the process directory."""
        return str(self.absolute())

    def absolute(self) -> Self:
        """Join the base; an absolute path is returned as is.
        `ViewPath.cwd()` is the logical working directory."""
        if self.is_absolute():
            return self
        return self.with_segments(self.base, self)

    def resolve(self, strict: bool = False) -> Path:  # type: ignore[override]
        """Leave the view: the physical path as a plain `Path`, for identity
        (cycles, duplicates), never for navigation or messages."""
        return Path(os.fspath(self)).resolve(strict)

    def relative_base(self) -> Self:
        """Directory that relative paths written inside this file resolve
        against: the real parent when the file itself is a symlink, else its
        parent in the view (a file reached through a linked folder stays in
        the view)."""
        if not self.is_symlink():
            return self.parent
        return self.with_segments(os.path.realpath(self)).parent

    def walk(
        self,
        top_down: bool = True,
        on_error: Callable[[OSError], object] | None = None,
        follow_symlinks: bool = False,
    ) -> Iterator[tuple[Self, list[str], list[str]]]:
        """`Path.walk`, with names sorted. A broken or looping link counts as
        a file. With `follow_symlinks` a real directory is never entered
        twice, so linked cycles end. Prune `dirnames` in place to skip
        folders. A directory that can't be listed goes to `on_error`, if
        given, and is skipped."""
        seen: set[tuple[int, int]] = set()
        if follow_symlinks and (root := _identity(self)) is not None:
            seen.add(root)
        yield from self._walk(top_down, on_error, follow_symlinks, seen)

    def _walk(
        self,
        top_down: bool,
        on_error: Callable[[OSError], object] | None,
        follow_symlinks: bool,
        seen: set[tuple[int, int]],
    ) -> Iterator[tuple[Self, list[str], list[str]]]:
        try:
            with os.scandir(self) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError as error:
            if on_error is not None:
                on_error(error)
            return
        dirnames: list[str] = []
        filenames: list[str] = []
        for entry in entries:
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            (dirnames if is_dir else filenames).append(entry.name)
        if top_down:
            yield self, dirnames, filenames
        for name in dirnames:
            child = self / name
            if follow_symlinks:
                identity = _identity(child)
                if identity is None or identity in seen:
                    continue
                seen.add(identity)
            elif child.is_symlink():
                continue
            yield from child._walk(top_down, on_error, follow_symlinks, seen)
        if not top_down:
            yield self, dirnames, filenames
