import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, PositiveInt, TypeAdapter, ValidationError
from rapidfuzz import fuzz as _fuzz

from tunery.fuzzy_index import FuzzyIndex, normalize_key
from tunery.parts import (
    BIG_BAND_DEFAULT_PART,
    LibraryType,
    TitleCandidate,
    chain_for_part,
    fuzzy_title_prefix,
    interpret_filename,
    longer_title_name,
    match_candidate,
    normalize_score_key,
    normalize_text,
    parse_part_name,
    rank_chain,
    rule_priority,
    stem_candidate,
)

MatchMode = Literal["exact", "fuzzy"]
PROJECT_CONFIG_NAME = "tunery.yaml"
PROJECT_CONFIG_NAMES = ("tunery.yaml", ".tunery.yaml")


class LibraryIndexEntry(BaseModel):
    """One part-index entry: page ranges of a title's parts inside its file."""

    title: str
    file: str | None = None
    parts: dict[
        str,
        PositiveInt | tuple[PositiveInt, PositiveInt] | list[tuple[PositiveInt, PositiveInt]],
    ]


def _index_ranges(
    value: PositiveInt | tuple[PositiveInt, PositiveInt] | list[tuple[PositiveInt, PositiveInt]],
) -> list[tuple[int, int]]:
    """One part value as [(page, length)]: `15` is one page, `[17, 3]` is one
    range, `[[26, 1]]` lists ranges."""
    if isinstance(value, int):
        return [(value, 1)]
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    ):
        return [(int(value[0]), int(value[1]))]
    return [(int(page), int(length)) for page, length in value]


@dataclass(frozen=True)
class LibraryMatch:
    title: str
    source: Path
    page: int | None = None
    length: int | None = None
    score: float | None = None
    # All page ranges of an index part, in list order (LIB7.3 renders them
    # all; page/length carry the first for now).
    ranges: list[tuple[int, int]] | None = None
    # `source` as reached through links (file-system view), for messages.
    view: Path | None = None

    @property
    def shown(self) -> Path:
        return self.view or self.source


def logical_cwd() -> Path:
    """Working directory without resolving symlinks.

    After `cd` through a symlink, `os.getcwd()` / `Path.cwd()` return the
    physical target. POSIX shells keep the logical path in `$PWD`; use that
    when it still names the current directory.
    """
    pwd = os.environ.get("PWD")
    if pwd:
        logical = Path(pwd)
        try:
            if logical.samefile("."):
                return logical
        except OSError:
            pass
    return Path.cwd()


def lexical_absolute(path: Path, base: Path | None = None) -> Path:
    """Absolute path joined to `base` (default: logical working directory),
    with `.`/`..` collapsed and symlinks kept."""
    if not path.is_absolute() and base is not None:
        path = base / path
    if not path.is_absolute():
        path = logical_cwd() / path
    return Path(os.path.normpath(path))


@dataclass(frozen=True)
class PartHit:
    """One requested part resolved to a chart."""

    requested: str
    step: str  # chain step that matched
    label: str  # the chart's own part label
    match: LibraryMatch


@dataclass
class TitleSelection:
    """Charts chosen for one title, with parts missing and warnings to show."""

    hits: list[PartHit]
    missing: list[str]
    warnings: list[str]


class Library(Protocol):
    def lookup_title(self, title: str) -> LibraryMatch | None: ...

    def lookup_file(self, file: str) -> LibraryMatch | None: ...


class DirectoryLibrary:
    def __init__(
        self,
        path: Path,
        *,
        match: MatchMode = "exact",
        library_type: LibraryType = "combo",
        index: list[LibraryIndexEntry] | None = None,
    ) -> None:
        # `path` as given is the library's place in the file-system view.
        self.view_path = lexical_absolute(path)
        self.path = path.resolve()
        self.match = match
        self.library_type = library_type
        self.index = list(index or [])
        if not self.path.is_dir():
            raise ValueError(f"Library directory not found: {self.view_path}")

        self._title_index = FuzzyIndex[Path]()
        self._file_index = FuzzyIndex[Path]()
        self._titles: dict[str, Path] = {}
        self._files: dict[str, Path] = {}
        self._folders: dict[Path, str | None] = {}
        # Resolved path -> path inside the library as discovered (links kept).
        self._display: dict[Path, Path] = {}
        discovered = sorted(self._discover(), key=lambda item: str(item[0]))
        self._items: list[tuple[Path, str | None]] = []
        for pdf_path, folder, display in discovered:
            self._items.append((pdf_path, folder))
            self._display[pdf_path] = display
        for pdf_path, folder in self._items:
            self._folders[pdf_path] = folder
            self._title_index.add(pdf_path.stem, pdf_path)
            self._file_index.add(pdf_path.name, pdf_path)
            self._titles.setdefault(normalize_key(pdf_path.stem), pdf_path)
            self._files.setdefault(normalize_key(pdf_path.name), pdf_path)
            if folder:
                self._title_index.add(folder, pdf_path)
                self._titles.setdefault(normalize_key(folder), pdf_path)

    @staticmethod
    def _is_transparent(name: str) -> bool:
        return name.startswith("_") and name != "_archive"

    def _discover(self) -> list[tuple[Path, str | None, Path]]:
        """Every PDF below the library: resolved path, innermost title folder,
        and path inside the library as reached (symlinks kept).

        Follows symlinks without revisiting a real directory. `_archive`
        subtrees are skipped; `_.*` folders are transparent (name ignored).
        """
        found: list[tuple[Path, str | None, Path]] = []
        visited: set[tuple[int, int]] = set()
        # Stack of (lexical dir path, title-folder chain from root).
        # Chain holds folder names from library root, transparent names omitted.
        try:
            root_stat = self.path.stat()
            visited.add((root_stat.st_dev, root_stat.st_ino))
        except OSError:
            pass
        stack: list[tuple[Path, list[str]]] = [(self.path, [])]
        while stack:
            current, chain = stack.pop()
            try:
                with os.scandir(current) as entries:
                    dirs: list[os.DirEntry[str]] = []
                    files: list[os.DirEntry[str]] = []
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=True):
                                dirs.append(entry)
                            elif entry.is_file(follow_symlinks=True):
                                files.append(entry)
                        except OSError:
                            continue
                    for entry in sorted(files, key=lambda e: e.name):
                        if entry.name.lower().endswith(".pdf"):
                            lexical = Path(current) / entry.name
                            # Innermost title folder = last non-transparent chain element.
                            folder = chain[-1] if chain else None
                            found.append(
                                (lexical.resolve(), folder, lexical.relative_to(self.path))
                            )
                    for entry in sorted(dirs, key=lambda e: e.name):
                        if entry.name == "_archive":
                            continue
                        try:
                            st = entry.stat(follow_symlinks=True)
                        except OSError:
                            continue
                        key = (st.st_dev, st.st_ino)
                        if key in visited:
                            continue
                        visited.add(key)
                        if self._is_transparent(entry.name):
                            stack.append((Path(current) / entry.name, chain))
                        else:
                            stack.append(
                                (Path(current) / entry.name, [*chain, entry.name])
                            )
            except OSError:
                continue
        return found

    def lookup_title(self, title: str) -> LibraryMatch | None:
        return self._lookup(title, self._titles, self._title_index, use_stem=True)

    def lookup_file(self, file: str) -> LibraryMatch | None:
        return self._lookup_file_plain(file)

    def _lookup_file_plain(self, file: str) -> LibraryMatch | None:
        path = Path(file)
        if path.is_absolute():
            return None
        candidate = (self.path / path).resolve()
        if (
            candidate.is_relative_to(self.path)
            and candidate.is_file()
            and candidate.suffix.lower() == ".pdf"
        ):
            return self._match(candidate.name, candidate)
        if path.parent != Path("."):
            return None
        return self._lookup(path.name, self._files, self._file_index, use_stem=False)

    def _lookup(
        self,
        query: str,
        exact_items: dict[str, Path],
        fuzzy_index: FuzzyIndex[Path],
        *,
        use_stem: bool,
    ) -> LibraryMatch | None:
        exact = exact_items.get(normalize_key(query))
        if exact is not None:
            return self._match(exact.stem if use_stem else exact.name, exact)
        if self.match == "exact":
            return None

        matches = fuzzy_index.match(query, score_cutoff=90, limit=1)
        if not matches:
            return None
        result = matches[0]
        return self._match(
            result.item.stem if use_stem else result.item.name, result.item, result.score
        )

    def display_path(self, path: Path) -> Path:
        """Path inside the library as discovered (through links)."""
        return self._display.get(path, path)

    def view_of(self, path: Path) -> Path:
        """A discovered file's path in the file-system view."""
        return self.view_path / self.display_path(path)

    def _match(self, title: str, path: Path, score: float | None = None) -> LibraryMatch:
        return LibraryMatch(title=title, source=path, score=score, view=self.view_of(path))

    def select_parts(
        self,
        title: str,
        parts: list[str],
        score_key: str | None = None,
        tags: list[str] | None = None,
    ) -> TitleSelection | None:
        """Choose a chart for each requested part of a title.

        Returns None when the library has no candidate for the title. One
        longer name is that title's file and is warned; several longer names
        leave the title not found and name them. Each part takes the best
        rank where any source (folder) has a candidate.
        Among those: candidates carrying all `tags`, then the fewest
        substitution steps, then the title's top source, then the fewest
        extra words, then path order. A part mapped by the chart's `part:`
        is requested as such, so the mapping is not a substitution. One warning per title reports several
        candidates for a part, or charts from more than one folder.
        """
        cands, name_warnings = self._titled_candidates(title)
        indexed, index_warnings = self.index_candidates(title, cands)
        cands = [*cands, *indexed]
        warnings: list[str] = [*name_warnings, *index_warnings]
        if not cands:
            # Several longer names are not this title: say so, and do not
            # report every requested part as missing.
            if name_warnings and not index_warnings:
                return TitleSelection([], [], warnings)
            return None if not warnings else TitleSelection([], list(parts), warnings)
        if score_key is not None:
            try:
                wanted = normalize_score_key(score_key)
            except ValueError:
                wanted = score_key
            keyed = [
                (p, c)
                for p, c in cands
                if c.score_key is not None and c.score_key.lower() == wanted.lower()
            ]
            if keyed:
                cands = keyed
            else:
                warnings.append(f'no {wanted} version of "{title}"')
        wanted_tags = [normalize_text(tag) for tag in tags or []]

        def carries_tags(cand: TitleCandidate) -> bool:
            words = {normalize_text(w) for w in [*cand.variations, *cand.free_tags]}
            return all(tag in words for tag in wanted_tags)

        if wanted_tags and not any(carries_tags(c) for _, c in cands):
            warnings.append(f'no {", ".join(tags or [])} version of "{title}"')
        source = {p: self.display_path(p).parent for p, _ in cands}
        top = _top_source(parts, cands, source)

        hits: list[PartHit] = []
        missing: list[str] = []
        others: dict[tuple[Path, tuple[Path, ...]], list[str]] = {}
        for part in parts:
            at_rank = _candidates_at_best_rank(part, cands)
            if not at_rank:
                missing.append(part)
                continue
            chain = chain_for_part(part)
            at_rank.sort(
                key=lambda h: (
                    not carries_tags(h[1]),
                    chain.index(h[2]),  # substitution steps
                    source[h[0]] != top,
                    h[1].tag_count,
                    str(self.display_path(h[0])),
                )
            )
            path, cand, step = at_rank[0]
            match = self._match(title, path)
            if cand.page is not None:
                match = replace(
                    match, page=cand.page, length=cand.length, ranges=cand.ranges
                )
            hits.append(PartHit(part, step, cand.part, match))
            if len(at_rank) > 1:
                rest = tuple(p for p, _, _ in at_rank[1:])
                others.setdefault((path, rest), []).append(part)
        if not hits:
            return TitleSelection([], list(parts), warnings)
        folders = sorted({str(source[h.match.source]) for h in hits})
        lines = [
            f"  {', '.join(names)}: {self.display_path(chosen)}"
            f" (also {', '.join(str(self.display_path(p)) for p in rest)})"
            for (chosen, rest), names in others.items()
        ]
        if len(folders) > 1:
            lines.append(f"  charts from {len(folders)} folders: {', '.join(folders)}")
        if lines:
            warnings.append("\n".join([f'warning "{title}":', *lines]))
        return TitleSelection(hits, missing, warnings)

    def index_candidates(
        self, title: str, cands: list[tuple[Path, TitleCandidate]]
    ) -> tuple[list[tuple[Path, TitleCandidate]], list[str]]:
        """Part-index entries naming `title` as candidates plus warnings.

        Each entry resolves its file (explicit `file:`, else the title's
        jumbo file) and contributes one candidate per part at that part's
        own step. An entry whose title finds no jumbo file, or several
        without `file:`, warns and counts its parts as missing.
        """
        pairs: list[tuple[Path, TitleCandidate]] = []
        warnings: list[str] = []
        entries = [
            entry
            for entry in self.index
            if normalize_key(entry.title) == normalize_key(title)
        ]
        if not entries and self.match == "fuzzy":
            best: LibraryIndexEntry | None = None
            best_score = 90
            for entry in self.index:
                score = _fuzz.ratio(
                    normalize_key(title), normalize_key(entry.title)
                )
                if score >= best_score:
                    best, best_score = entry, score
            entries = [best] if best is not None else []
        for entry in entries:
            if entry.file is not None:
                raw = Path(entry.file)
                resolved = raw if raw.is_absolute() else self.path / raw
                if not (resolved.is_file() and resolved.suffix.lower() == ".pdf"):
                    raise ValueError(
                        f'Index file not found for "{title}": {entry.file}'
                    )
                source = resolved.resolve()
                base = interpret_filename(source.stem)
            else:
                jumbos = (
                    sorted(
                        {str(p) for p, c in cands if c.part == BIG_BAND_DEFAULT_PART},
                    )
                    if rule_priority("jumbo default", self.library_type) > 0
                    else []
                )
                if len(jumbos) != 1:
                    warnings.append(
                        f'no jumbo file for "{title}"'
                        if not jumbos
                        else f'several jumbo files for "{title}": '
                        + ", ".join(Path(p).name for p in jumbos)
                    )
                    continue
                source = Path(jumbos[0])
                base = next(c for p, c in cands if str(p) == jumbos[0])
            for name, value in entry.parts.items():
                part = parse_part_name(name, title)
                ranges = _index_ranges(value)
                pairs.append(
                    (
                        source,
                        TitleCandidate(
                            remainder="",
                            parts=[part],
                            part=part,
                            score_key=base.score_key,
                            variations=list(base.variations),
                            free_tags=list(base.free_tags),
                            rules_fired=[*base.rules_fired, "index entry"],
                            noise=list(base.noise),
                            page=ranges[0][0],
                            length=ranges[0][1],
                            ranges=ranges,
                        ),
                    )
                )
        return pairs, warnings

    def all_candidates(self, title: str) -> list[tuple[Path, TitleCandidate]]:
        """All title-matching candidates with interpretation, path order."""
        cands, _warnings = self._titled_candidates(title)
        return cands

    def _titled_candidates(
        self, title: str
    ) -> tuple[list[tuple[Path, TitleCandidate]], list[str]]:
        """Ordinary candidates, or one longer file name when that is all.

        A file whose name starts with the title and continues with more
        words is the title's file only when no ordinary candidate exists and
        that name is the only one. The extra words stay part of the title,
        so the file is the title's untagged chart. Several such names match
        nothing and are listed.
        """
        ordinary = self._ordinary_candidates(title)
        if ordinary:
            return ordinary, []
        groups = self._longer_file_names(title)
        if not groups:
            return [], []
        if len(groups) > 1:
            listed = ", ".join(f'"{name}"' for name in sorted(groups))
            return [], [f'several names start with "{title}": {listed}']
        name, items = next(iter(groups.items()))
        cands = [
            (path, cand)
            for path, folder in items
            if (cand := match_candidate(title, title, folder, self.library_type))
            is not None
        ]
        if not cands:
            return [], []
        return cands, [f'matched "{title}" to "{name}"']

    def _longer_file_names(
        self, title: str
    ) -> dict[str, list[tuple[Path, str | None]]]:
        """File stems that start with `title` plus more words, path order."""
        groups: dict[str, list[tuple[Path, str | None]]] = {}
        for pdf_path, folder in self._items:
            stem = pdf_path.stem
            if longer_title_name(title, stem):
                groups.setdefault(stem, []).append((pdf_path, folder))
        return groups

    def _ordinary_candidates(self, title: str) -> list[tuple[Path, TitleCandidate]]:
        """Exact prefix matches, then fuzzy word prefixes when the library
        asks for them and nothing matched exactly.

        What follows the matched prefix is tags.
        """
        out: list[tuple[Path, TitleCandidate]] = []
        for pdf_path, folder in self._items:
            cand = match_candidate(
                title, pdf_path.stem, folder, self.library_type
            )
            if cand is not None:
                out.append((pdf_path, cand))
        if out or self.match != "fuzzy":
            return out
        for pdf_path, folder in self._items:
            for text in (pdf_path.stem, folder):
                prefix = fuzzy_title_prefix(title, text) if text else None
                if prefix is None:
                    continue
                cand = match_candidate(
                    prefix, pdf_path.stem, folder, self.library_type
                )
                if cand is not None:
                    out.append((pdf_path, cand))
                    break
        return out


class ScoreFolder(DirectoryLibrary):
    """A chart's `file:` folder: every PDF below it belongs to the chart's
    title, so a name that doesn't match the title is read whole as tags."""

    def _titled_candidates(
        self, title: str
    ) -> tuple[list[tuple[Path, TitleCandidate]], list[str]]:
        return self.all_candidates(title), []

    def all_candidates(self, title: str) -> list[tuple[Path, TitleCandidate]]:
        return [
            (
                path,
                match_candidate(title, path.stem, folder or self.view_path.name)
                or stem_candidate(path.stem, folder or self.view_path.name),
            )
            for path, folder in self._items
        ]


def _matched_step(cand: TitleCandidate, rank: list[str]) -> str | None:
    # Jumbo and combined-set scores are listed but never matched.
    if not cand.selectable:
        return None
    for step in rank:
        if step in cand.parts or step == cand.part:
            return step
    return None


def _candidates_at_best_rank(
    part: str, cands: list[tuple[Path, TitleCandidate]]
) -> list[tuple[Path, TitleCandidate, str]]:
    """Candidates at the first rank of the part's chain that has any."""
    for rank in rank_chain(part):
        at_rank = [
            (path, cand, step)
            for path, cand in cands
            if (step := _matched_step(cand, rank)) is not None
        ]
        if at_rank:
            return at_rank
    return []


def _top_source(
    parts: list[str],
    cands: list[tuple[Path, TitleCandidate]],
    source: dict[Path, Path],
) -> Path:
    """Folder with the most requested parts at their own step, then the
    fewest extra words, then path order."""
    own: dict[Path, int] = {}
    extra: dict[Path, int] = {}
    for path, cand in cands:
        folder = source[path]
        extra[folder] = min(extra.get(folder, cand.tag_count), cand.tag_count)
        own.setdefault(folder, 0)
    for part in parts:
        for folder in {
            source[path] for path, cand in cands if part in cand.parts or part == cand.part
        }:
            own[folder] += 1
    return min(extra, key=lambda folder: (-own[folder], extra[folder], str(folder)))


class IndexedTune(BaseModel):
    title: str
    page: PositiveInt
    pages: PositiveInt = 1


INDEXED_TUNES_ADAPTER = TypeAdapter(list[IndexedTune])


class IndexedPdfLibrary:
    def __init__(
        self,
        index_path: Path,
        source: Path,
        tunes: list[IndexedTune],
        *,
        shift: int = 0,
        match: MatchMode = "exact",
    ) -> None:
        self.index_path = index_path.resolve()
        self.source_view = lexical_absolute(source)
        self.source = source.resolve()
        self.match = match
        self._tunes: dict[str, IndexedTune] = {}
        self._title_index = FuzzyIndex[IndexedTune]()
        self._shift = shift
        for tune in tunes:
            effective_page = tune.page + shift
            if effective_page < 1:
                raise ValueError(
                    f'Invalid page for "{tune.title}" in {self.index_path}: '
                    f"{tune.page} + shift {shift} must be positive"
                )
            self._tunes.setdefault(normalize_key(tune.title), tune)
            self._title_index.add(tune.title, tune)

    @classmethod
    def from_json(
        cls,
        index_path: Path,
        source: Path,
        *,
        shift: int = 0,
        match: MatchMode = "exact",
    ) -> "IndexedPdfLibrary":
        resolved_index = index_path.resolve()
        try:
            raw_data = json.loads(resolved_index.read_text(encoding="utf-8"))
            tunes = INDEXED_TUNES_ADAPTER.validate_python(raw_data)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError(
                f"Invalid indexed PDF library {resolved_index}: {error}"
            ) from error

        if not source.resolve().is_file():
            raise ValueError(
                f"Source PDF not found: {source} (configured for {index_path})"
            )
        if source.suffix.lower() != ".pdf":
            raise ValueError(
                f"Source must be a PDF: {source} (configured for {index_path})"
            )
        return cls(
            resolved_index,
            source,
            tunes,
            shift=shift,
            match=match,
        )

    def lookup_title(self, title: str) -> LibraryMatch | None:
        tune = self._tunes.get(normalize_key(title))
        score: float | None = None
        if tune is None and self.match == "fuzzy":
            matches = self._title_index.match(title, score_cutoff=90, limit=1)
            if matches:
                tune = matches[0].item
                score = matches[0].score
        if tune is None:
            return None
        return LibraryMatch(
            title=tune.title,
            source=self.source,
            page=tune.page + self._shift,
            length=tune.pages,
            score=score,
            view=self.source_view,
        )

    def lookup_file(self, file: str) -> LibraryMatch | None:
        return None


class LibraryCollection:
    def __init__(
        self, libraries: list[Library] | None = None, config_path: Path | None = None
    ) -> None:
        self._libraries = list(libraries or [])
        # Project configuration the collection was loaded from, if any.
        self.config_path = config_path

    @property
    def directories(self) -> list[Path]:
        result: list[Path] = []
        for library in self._libraries:
            if isinstance(library, DirectoryLibrary):
                result.append(library.path)
            elif isinstance(library, LibraryCollection):
                result.extend(library.directories)
        return result

    def add(self, library: Library) -> None:
        self._libraries.append(library)

    def lookup_title(self, title: str) -> LibraryMatch | None:
        for library in reversed(self._libraries):
            result = library.lookup_title(title)
            if result is not None:
                return result
        return None

    def lookup_file(self, file: str) -> LibraryMatch | None:
        for library in reversed(self._libraries):
            result = library.lookup_file(file)
            if result is not None:
                return result
        return None

    def select_parts(
        self,
        title: str,
        parts: list[str],
        score_key: str | None = None,
        tags: list[str] | None = None,
    ) -> TitleSelection | None:
        """Choose charts for a title's parts, highest-priority library first.

        The first library with a chart on any requested part's chain wins the
        title; parts it lacks are missing, never taken from a lower library,
        so one title's parts belong to one arrangement. Indexed libraries
        answer every part with their concert pages. Returns None when no
        library has the title at all.
        """
        found: TitleSelection | None = None
        for library in reversed(self._libraries):
            if isinstance(library, (LibraryCollection, DirectoryLibrary)):
                selection = library.select_parts(title, parts, score_key, tags)
            else:
                match = library.lookup_title(title)
                selection = (
                    TitleSelection(
                        [PartHit(part, "Concert", "Concert", match) for part in parts],
                        [],
                        [],
                    )
                    if match is not None
                    else None
                )
            if selection is None:
                continue
            if selection.hits:
                return selection
            found = found or selection
        return found

    def all_candidates(self, title: str) -> list[tuple[str, Path, object | None]]:
        """All candidates per library for -vv, highest priority first."""
        out: list[tuple[str, Path, object | None]] = []
        for library in reversed(self._libraries):
            if isinstance(library, LibraryCollection):
                out.extend(library.all_candidates(title))
            elif isinstance(library, DirectoryLibrary):
                named = library.all_candidates(title)
                indexed, _ = library.index_candidates(title, named)
                for path, cand in [*named, *indexed]:
                    out.append((str(library.view_of(path).parent), path, cand))
            else:
                match = library.lookup_title(title)
                if match is not None:
                    out.append(("indexed", match.source, None))
        return out


class LibraryRecord(BaseModel):
    library: str
    source: str | None = None
    shift: int | None = None
    match: MatchMode = "exact"
    type: LibraryType = "combo"
    index: list[LibraryIndexEntry] | None = None

    def _load_collection(self, path: Path, seen: tuple[Path, ...]) -> Library:
        if "source" in self.model_fields_set:
            raise ValueError(f"Library collection {path} must not specify source")
        if "shift" in self.model_fields_set:
            raise ValueError(f"Library collection {path} must not specify shift")
        if path in seen:
            chain = " -> ".join(str(p) for p in (*seen, path))
            raise ValueError(f"Circular library collection include: {chain}")
        try:
            raw_collection = yaml.safe_load(path.read_text(encoding="utf-8"))
            records = [
                LibraryRecord.model_validate(item) for item in (raw_collection or [])
            ]
        except (OSError, yaml.YAMLError, ValidationError, TypeError) as error:
            raise ValueError(
                f"Invalid library collection {path}: {error}"
            ) from error
        if "match" in self.model_fields_set:
            records = [
                record
                if "match" in record.model_fields_set
                else record.model_copy(update={"match": self.match})
                for record in records
            ]
        if "type" in self.model_fields_set:
            records = [
                record
                if "type" in record.model_fields_set
                else record.model_copy(update={"type": self.type})
                for record in records
            ]
        seen = (*seen, path)
        return LibraryCollection(
            [record.load(path.parent, seen) for record in records]
        )

    def load(self, base_dir: Path, _seen: tuple[Path, ...] = ()) -> Library:
        """Load the record; relative paths join `base_dir` lexically, so a
        path through a link stays in the file-system view."""
        path = lexical_absolute(Path(self.library), base_dir)
        if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
            # A symlinked collection file resolves its paths against its real
            # location; otherwise they stay in the file-system view, even
            # through a linked folder.
            return self._load_collection(
                path.resolve() if path.is_symlink() else path, _seen
            )
        if path.is_dir():
            if "source" in self.model_fields_set:
                raise ValueError(
                    f"Directory library {path} must not specify source"
                )
            if "shift" in self.model_fields_set:
                raise ValueError(f"Directory library {path} must not specify shift")
            return DirectoryLibrary(
                path, match=self.match, library_type=self.type, index=self.index
            )
        if path.suffix.lower() == ".json" and path.is_file():
            if self.source is None:
                raise ValueError(f"Indexed PDF library {path} requires source")
            source = lexical_absolute(Path(self.source), base_dir)
            shift = 0 if self.shift is None else self.shift
            return IndexedPdfLibrary.from_json(
                path,
                source,
                shift=shift,
                match=self.match,
            )
        raise ValueError(
            f"Library path must be a directory, JSON file, or YAML file: {path}"
        )


def find_project_config(start_dir: Path) -> Path | None:
    """Walk up lexically from start_dir to the root looking for project config.

    Checks `tunery.yaml` and `.tunery.yaml` in each directory. The first
    directory holding one wins; holding both is an error. Symlinks are not
    resolved, so the walk climbs back through links into the base hierarchy.
    """
    directory = lexical_absolute(start_dir)
    while True:
        plain = directory / PROJECT_CONFIG_NAMES[0]
        dotted = directory / PROJECT_CONFIG_NAMES[1]
        has_plain = plain.is_file()
        has_dotted = dotted.is_file()
        if has_plain and has_dotted:
            raise ValueError(
                f"Both {PROJECT_CONFIG_NAMES[0]} and {PROJECT_CONFIG_NAMES[1]} "
                f"present in {directory}"
            )
        if has_plain:
            return plain
        if has_dotted:
            return dotted
        if directory.parent == directory:
            return None
        directory = directory.parent


def load_project_libraries(start_dir: Path | None = None) -> LibraryCollection:
    """Libraries of the first project config found walking up from `start_dir`
    (default: logical working directory); without one, `start_dir` itself."""
    start = lexical_absolute(start_dir or logical_cwd())
    config_path = find_project_config(start)
    if config_path is None:
        print("Project configuration was not found, using defaults")
        return LibraryCollection([DirectoryLibrary(start)])
    # A symlinked config file resolves its paths against its real location;
    # otherwise they join its directory in the file-system view.
    base_dir = config_path.resolve().parent if config_path.is_symlink() else config_path.parent
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        records = [LibraryRecord.model_validate(item) for item in (raw_config or [])]
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as error:
        raise ValueError(f"Invalid project configuration {config_path}: {error}") from error
    return LibraryCollection([record.load(base_dir) for record in records], config_path)
