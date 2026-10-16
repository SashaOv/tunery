import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, PositiveInt, TypeAdapter, ValidationError

from tunery.fuzzy_index import FuzzyIndex, normalize_key
from tunery.keys import CONCERT_SUFFIX, is_concert, keyed_filename, normalize_key_name
from tunery.parts import (
    TitleCandidate,
    fuzzy_title_prefix,
    match_candidate,
    rank_chain,
)

MatchMode = Literal["exact", "fuzzy"]
PROJECT_CONFIG_NAME = "tunery.yaml"
PROJECT_CONFIG_NAMES = ("tunery.yaml", ".tunery.yaml")
PROJECT_CONFIG_PARENT_LIMIT = 8


@dataclass(frozen=True)
class LibraryMatch:
    title: str
    source: Path
    page: int | None = None
    length: int | None = None
    score: float | None = None


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
    def lookup_title(
        self, title: str, key: str | None = None
    ) -> LibraryMatch | None: ...

    def lookup_file(
        self, file: str, key: str | None = None
    ) -> LibraryMatch | None: ...


class DirectoryLibrary:
    def __init__(self, path: Path, *, match: MatchMode = "exact") -> None:
        self.path = path.resolve()
        self.match = match
        if not self.path.is_dir():
            raise ValueError(f"Library directory not found: {self.path}")

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

    def lookup_title(
        self, title: str, key: str | None = None
    ) -> LibraryMatch | None:
        for variant in self._key_variants(title, key):
            result = self._lookup(
                variant, self._titles, self._title_index, use_stem=True
            )
            if result is not None:
                return result
        return None

    def lookup_file(
        self, file: str, key: str | None = None
    ) -> LibraryMatch | None:
        candidates = [file]
        keyed = keyed_filename(file, key) if key is not None else None
        if keyed is not None:
            if is_concert(key):  # type: ignore[arg-type]
                candidates = [keyed, file]
            else:
                candidates = [keyed]
        for candidate in candidates:
            result = self._lookup_file_plain(candidate)
            if result is not None:
                return result
        return None

    @staticmethod
    def _key_variants(title: str, key: str | None) -> list[str]:
        """Title query variants for a key edition, most specific first."""
        if key is None:
            return [title]
        if is_concert(key):
            return [f"{title}.{CONCERT_SUFFIX}", title]
        return [f"{title}.{key}"]

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
            return LibraryMatch(title=candidate.name, source=candidate)
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
            return LibraryMatch(
                title=exact.stem if use_stem else exact.name,
                source=exact,
            )
        if self.match == "exact":
            return None

        matches = fuzzy_index.match(query, score_cutoff=90, limit=1)
        if not matches:
            return None
        result = matches[0]
        return LibraryMatch(
            title=result.item.stem if use_stem else result.item.name,
            source=result.item,
            score=result.score,
        )

    def display_path(self, path: Path) -> Path:
        """Path inside the library as discovered (through links)."""
        return self._display.get(path, path)

    def select_parts(
        self, title: str, parts: list[str], score_key: str | None = None
    ) -> TitleSelection | None:
        """Choose a chart for each requested part of a title.

        Returns None when the library has no candidate for the title. Each
        part takes the best rank where any source (folder) has a candidate.
        Among those: the title's top source first, then the fewest extra
        words, then path order. One warning per title reports several
        candidates for a part, or charts from more than one folder.
        """
        cands = self.all_candidates(title)
        if not cands:
            return None
        warnings: list[str] = []
        if score_key is not None:
            try:
                wanted = normalize_key_name(score_key)
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
            at_rank.sort(
                key=lambda h: (
                    source[h[0]] != top,
                    h[1].tag_count,
                    str(self.display_path(h[0])),
                )
            )
            path, cand, step = at_rank[0]
            hits.append(
                PartHit(part, step, cand.part, LibraryMatch(title=title, source=path))
            )
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

    def all_candidates(self, title: str) -> list[tuple[Path, TitleCandidate]]:
        """All title-matching candidates with interpretation, path order.

        Exact prefix matches first; for fuzzy libraries with none, match the
        title against word prefixes of each name text and interpret what
        follows the matched prefix as tags.
        """
        out: list[tuple[Path, TitleCandidate]] = []
        for pdf_path, folder in self._items:
            cand = match_candidate(title, pdf_path.stem, folder)
            if cand is not None:
                out.append((pdf_path, cand))
        if out or self.match != "fuzzy":
            return out
        for pdf_path, folder in self._items:
            for text in (pdf_path.stem, folder):
                prefix = fuzzy_title_prefix(title, text) if text else None
                if prefix is None:
                    continue
                cand = match_candidate(prefix, pdf_path.stem, folder)
                if cand is not None:
                    out.append((pdf_path, cand))
                    break
        return out


def _matched_step(cand: TitleCandidate, rank: list[str]) -> str | None:
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

        resolved_source = source.resolve()
        if not resolved_source.is_file():
            raise ValueError(
                f"Source PDF not found: {resolved_source} "
                f"(configured for {resolved_index})"
            )
        if resolved_source.suffix.lower() != ".pdf":
            raise ValueError(
                f"Source must be a PDF: {resolved_source} "
                f"(configured for {resolved_index})"
            )
        return cls(
            resolved_index,
            resolved_source,
            tunes,
            shift=shift,
            match=match,
        )

    def lookup_title(
        self, title: str, key: str | None = None
    ) -> LibraryMatch | None:
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
        )

    def lookup_file(
        self, file: str, key: str | None = None
    ) -> LibraryMatch | None:
        return None


class LibraryCollection:
    def __init__(self, libraries: list[Library] | None = None) -> None:
        self._libraries = list(libraries or [])

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

    def lookup_title(
        self, title: str, key: str | None = None
    ) -> LibraryMatch | None:
        for library in reversed(self._libraries):
            result = library.lookup_title(title, key)
            if result is not None:
                return result
        return None

    def lookup_file(
        self, file: str, key: str | None = None
    ) -> LibraryMatch | None:
        for library in reversed(self._libraries):
            result = library.lookup_file(file, key)
            if result is not None:
                return result
        return None

    def select_parts(
        self, title: str, parts: list[str], score_key: str | None = None
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
                selection = library.select_parts(title, parts, score_key)
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
                for path, cand in library.all_candidates(title):
                    out.append((str(library.path), path, cand))
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
        seen = (*seen, path)
        return LibraryCollection(
            [record.load(path.parent, seen) for record in records]
        )

    def load(self, base_dir: Path, _seen: tuple[Path, ...] = ()) -> Library:
        path = Path(self.library)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
            return self._load_collection(path, _seen)
        if path.is_dir():
            if "source" in self.model_fields_set:
                raise ValueError(
                    f"Directory library {path} must not specify source"
                )
            if "shift" in self.model_fields_set:
                raise ValueError(f"Directory library {path} must not specify shift")
            return DirectoryLibrary(path, match=self.match)
        if path.suffix.lower() == ".json" and path.is_file():
            if self.source is None:
                raise ValueError(f"Indexed PDF library {path} requires source")
            source = Path(self.source)
            if not source.is_absolute():
                source = (base_dir / source).resolve()
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


def find_project_config(
    start_dir: Path, limit: int | None = PROJECT_CONFIG_PARENT_LIMIT
) -> Path | None:
    """Walk up lexically from start_dir looking for project config.

    Checks `tunery.yaml` and `.tunery.yaml` in each directory. The first
    directory holding one wins; holding both is an error. `limit` caps how
    many parents to climb (None walks to the filesystem root). Symlinks in
    intermediate components are not resolved.
    """
    directory = Path(os.path.normpath(str(start_dir)))
    depth = 0
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
        if limit is not None and depth >= limit:
            break
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
        depth += 1
    return None


def _logical_cwd() -> Path:
    """Working directory without resolving symlinks (see render.logical_cwd)."""
    pwd = os.environ.get("PWD")
    if pwd:
        logical = Path(pwd)
        try:
            if logical.samefile("."):
                return logical
        except OSError:
            pass
    return Path.cwd()


def _lexical_absolute(path: Path) -> Path:
    if not path.is_absolute():
        path = _logical_cwd() / path
    return Path(os.path.normpath(str(path)))


def load_library_records(
    records: list[LibraryRecord], base_dir: Path
) -> LibraryCollection:
    return LibraryCollection([record.load(base_dir) for record in records])


def load_project_libraries(
    start_dir: Path | None = None, limit: int | None = PROJECT_CONFIG_PARENT_LIMIT
) -> LibraryCollection:
    if start_dir is None:
        start = _logical_cwd()
    else:
        start = Path(start_dir)
        if not start.is_absolute():
            start = _lexical_absolute(start)
    config_path = find_project_config(start, limit)
    if config_path is None:
        print("Project configuration was not found, using defaults")
        return LibraryCollection([DirectoryLibrary(start)])
    # Dereference symlinks so relative library/source paths resolve to real files.
    config_path = config_path.resolve()

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        records = [LibraryRecord.model_validate(item) for item in (raw_config or [])]
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as error:
        raise ValueError(f"Invalid project configuration {config_path}: {error}") from error
    return load_library_records(records, config_path.parent)
