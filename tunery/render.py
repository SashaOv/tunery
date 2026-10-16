"""PDF rendering logic for combining PDFs with table of contents."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PositiveInt, RootModel, ValidationError

from tunery.composer import Composer
from tunery.keys import (
    CONCERT_KEY,
    expand_keys,
    is_concert,
    keyed_filename,
    output_path_for_key,
)
from tunery.library import (
    PROJECT_CONFIG_PARENT_LIMIT,
    LibraryCollection,
    LibraryRecord,
    load_project_libraries,
)
from tunery.parts import canonical_part_name, normalize_text


class FileEntry(BaseModel):
    """A file entry specifying a PDF source and optional page range/notes.

    Either 'file' or 'title' must be provided. If only 'title' is provided,
    the file will be looked up in configured libraries.
    """

    file: str | None = None
    title: str | None = None
    page: PositiveInt | None = None
    length: PositiveInt | None = None
    notes: str | None = None
    key: str | None = None

    def model_post_init(self, context: object, /) -> None:
        if not self.file and not self.title:
            raise ValueError("Either 'file' or 'title' must be provided")


class SectionEntry(BaseModel):
    """A section containing nested file entries or subsections."""

    section: str
    body: list["FileEntry | SectionEntry"] = Field(default_factory=list)


class ConfigEntry(BaseModel):
    """Configuration entry adding libraries for subsequent lookup."""

    config: list[LibraryRecord]


class Layout(RootModel[list[ConfigEntry | SectionEntry | FileEntry]]):
    """The complete layout schema - a list of config, sections, and/or file entries."""



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


def lexical_absolute(path: Path) -> Path:
    """Absolute path with `.`/`..` collapsed and intermediate symlinks kept."""
    if not path.is_absolute():
        path = logical_cwd() / path
    return Path(os.path.normpath(path))


def output_path_for_part(output: Path, part: str | list[str] | None) -> Path:
    """Derive per-part setbook output path from the base output path."""
    if part is None:
        return output
    if isinstance(part, list):
        if len(part) == 1:
            return output_path_for_part(output, part[0])
        canons = [canonical_part_name(p) for p in part]
        return output.with_name(f"{output.stem}.{'+'.join(canons)}{output.suffix}")
    canon = canonical_part_name(part)
    if canon in ("Concert", "C"):
        return output
    if output.stem == canon or output.stem.endswith(f".{canon}"):
        return output
    return output.with_name(f"{output.stem}.{canon}{output.suffix}")


def output_path_for_label(output: Path, label: str | None) -> Path:
    """Derive output path from a part-spec label (RS stays .RS.pdf)."""
    if label is None or label in ("Concert", "C"):
        return output
    # Don't double-suffix when -o already ends with the part label.
    if output.stem == label or output.stem.endswith(f".{label}"):
        return output
    return output.with_name(f"{output.stem}.{label}{output.suffix}")


def expand_part_specs(parts: list[str] | str | None) -> list[tuple[str, list[str]]]:
    """Normalize --part options into (output label, subparts) per output.

    Single RS/Rhythm/Rhythm Section expands to Piano,Guitar,Bass,Drums with
    output label RS/Rhythm/Rhythm Section respectively. Comma-separated values
    combine subparts with '+' output naming.
    """
    if parts is None:
        return [("Concert", ["Concert"])]
    if isinstance(parts, str):
        parts = [parts]
    specs: list[tuple[str, list[str]]] = []
    for spec in parts:
        raw_subs = [s.strip() for s in spec.split(",") if s.strip()]
        if not raw_subs:
            specs.append(("Concert", ["Concert"]))
            continue
        if len(raw_subs) == 1:
            norm = normalize_text(raw_subs[0])
            if norm in ("rs", "rhythm", "rhythm section"):
                label = {"rs": "RS", "rhythm": "Rhythm", "rhythm section": "Rhythm Section"}[norm]
                specs.append((label, ["Piano", "Guitar", "Bass", "Drums"]))
                continue
        canons = [canonical_part_name(s) for s in raw_subs]
        if len(canons) == 1:
            specs.append((canons[0], canons))
        else:
            specs.append(("+".join(canons), canons))
    if not specs:
        return [("Concert", ["Concert"])]
    return specs


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """
    Resolve a file path relative to base_dir if it's relative, otherwise return as-is.

    Normalizes lexically without dereferencing intermediate symlinks, so `..`
    climbs above symlinked parents instead of their targets.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return lexical_absolute(base_dir / path)


def resolve_base_dir(
    layout_path: Path | None, explicit_base: Path | None
) -> tuple[Path, int | None]:
    """Choose the config search base and parent-walk limit.

    Explicit `--base` wins. Otherwise a relative layout path (or no layout)
    searches from the logical working directory with at most eight parents;
    an absolute layout path searches from its directory up to the root.
    """
    if explicit_base is not None:
        return lexical_absolute(explicit_base), None
    if layout_path is not None and layout_path.is_absolute():
        return lexical_absolute(layout_path.parent), None
    return logical_cwd(), PROJECT_CONFIG_PARENT_LIMIT


@dataclass
class ProcessEntryResult:
    """Base class for processing results."""
    title: str

    def format(self) -> str:
        """Format the result as a human-readable string."""
        raise NotImplementedError


@dataclass
class NotFoundResult(ProcessEntryResult):
    """Title not found."""
    hint: str | None = None  # e.g., 'Is this "Similar Title" in "Book"?'

    def format(self) -> str:
        msg = f'not found "{self.title}"'
        if self.hint:
            msg += f'. {self.hint}'
        return msg


def format_source_location(source_path: Path, page: int | None = None) -> str:
    """Format a match location: library directory, source stem, and page if available."""
    details = source_path.stem
    if page is not None:
        details += f", page {page}"
    return f'"{source_path.parent}" ({details})'


@dataclass
class SuccessResult(ProcessEntryResult):
    """Successfully found and processed a file entry."""
    page: int
    source_path: Path
    source_page: int | None = None  # starting page within the source PDF
    matched_title: str | None = None  # set if fuzzy matched
    score: float | None = None  # set if fuzzy matched

    def format(self) -> str:
        location = format_source_location(self.source_path, self.source_page)
        if self.matched_title is not None:
            return f'matched  "{self.title}" with "{self.matched_title}" from {location} ({self.score:.0f}%)'
        else:
            return f'found    "{self.title}" in {location}'


def display_title(
    explicit: str | None, resolved: Path, key: str | None
) -> str:
    """Display title for an entry, without any key edition suffix."""
    if explicit:
        return explicit
    stem = resolved.stem
    if key is not None:
        suffix = f".{key}" if not is_concert(key) else ".Concert"
        if stem.lower().endswith(suffix.lower()):
            return stem[: -len(suffix)]
    return stem


def process_file_entry(
    entry: FileEntry,
    default_dir: Path,
    composer: Composer,
    libraries: LibraryCollection | None = None,
    key: str = CONCERT_KEY,
    part: str | list[str] = "Concert",
    verbose: int = 0,
) -> ProcessEntryResult:
    """
    Add the requested pages for an entry to the combined PDF.

    `part` is the subpart list for one output (comma-combined). File entries
    ignore parts. Title entries resolve each subpart via chains, dedupe same
    files, report missing parts, and nest bookmarks when several files result.
    """

    matched_title: str | None = None
    score: float | None = None
    subparts: list[str] = (
        [part] if isinstance(part, str) else list(part) if part else ["Concert"]
    )
    if not subparts:
        subparts = ["Concert"]
    # `file:` entries with explicit page selections address the named PDF
    # directly and skip key edition handling; everything else resolves the
    # per-key edition source.
    lookup_key = (
        key
        if not (
            entry.file and (entry.page is not None or entry.length is not None)
        )
        else None
    )
    if entry.file:
        library_match = (
            libraries.lookup_file(entry.file, lookup_key) if libraries else None
        )
        if library_match is not None:
            input_pdf_path = library_match.source
        else:
            keyed = (
                keyed_filename(entry.file, lookup_key)
                if lookup_key is not None
                else None
            )
            if keyed is not None:
                keyed_path = resolve_path(keyed, default_dir)
                if keyed_path.is_file():
                    input_pdf_path = keyed_path
                elif lookup_key is not None and is_concert(lookup_key):
                    input_pdf_path = resolve_path(entry.file, default_dir)
                else:
                    return NotFoundResult(
                        title=entry.title or Path(entry.file).stem
                    )
            else:
                input_pdf_path = resolve_path(entry.file, default_dir)
        title = display_title(entry.title, input_pdf_path, lookup_key)
        page = entry.page
        length = entry.length
        entry_page = composer.add(
            title=title,
            source=input_pdf_path,
            start=page,
            pages=length,
            notes=entry.notes,
        )
        return SuccessResult(
            title=title,
            page=entry_page,
            source_path=input_pdf_path,
            source_page=page,
            matched_title=matched_title,
            score=score,
        )
    if not entry.title:
        raise ValueError("Entry must have either 'file' or 'title'")
    title = entry.title
    # Non-concert -k selects with no substitution (keys semantics).
    if lookup_key is not None and not is_concert(lookup_key):
        library_match = (
            libraries.lookup_title(title, lookup_key) if libraries else None
        )
        if library_match is None:
            return NotFoundResult(title=title)
        input_pdf_path = library_match.source
        page = entry.page if entry.page is not None else library_match.page
        length = entry.length if entry.length is not None else library_match.length
        if library_match.score is not None:
            matched_title = library_match.title
            score = library_match.score
        entry_page = composer.add(
            title=title, source=input_pdf_path, start=page, pages=length, notes=entry.notes
        )
        return SuccessResult(
            title=title,
            page=entry_page,
            source_path=input_pdf_path,
            source_page=page,
            matched_title=matched_title,
            score=score,
        )
    if libraries is None:
        return NotFoundResult(title=title)
    # Title exists if any candidate (or indexed hit) matches.
    existing = libraries.all_candidates(title)
    if not existing:
        return NotFoundResult(title=title)
    locked = libraries.select_subparts(title, subparts, entry.key)
    if locked is None:
        return NotFoundResult(title=title)
    locked_hits, locked_missing = locked
    for sub in locked_missing:
        print(f'missing {sub} for "{title}"')
    hits: list[tuple[str, str, str, object]] = []  # (subpart, step, label, match)
    for requested_canon, step, label, match in locked_hits:
        hits.append((requested_canon, step, label, match))
        if verbose >= 1:
            print(f'  "{title}" {requested_canon}: {match.source.name}')
            if step != requested_canon:
                print(f"  {requested_canon} → {step}")
    if verbose >= 2:
        for lib_desc, path, cand in existing:
            if cand is None:
                print(f'  candidate "{title}" in "{lib_desc}" ({path.name}): indexed')
            else:
                print(
                    f'  candidate "{title}" in "{lib_desc}" ({path.name}): '
                    f"part={cand.part}, key={cand.score_key}, "
                    f"var={cand.variations}, free={cand.free_tags}, "
                    f"rules={cand.rules_fired}"
                )
    if not hits:
        return NotFoundResult(title=title)
    # Dedupe same file, preserving first subpart order.
    seen: dict[Path, tuple[str, str, str, object]] = {}
    for requested_canon, step, label, match in hits:
        src = match.source.resolve() if hasattr(match.source, "resolve") else match.source
        if src not in seen:
            seen[src] = (requested_canon, step, label, match)
    unique = list(seen.values())
    if len(unique) == 1:
        _requested, _step, _label, match = unique[0]
        page = entry.page if entry.page is not None else match.page
        length = entry.length if entry.length is not None else match.length
        entry_page = composer.add(
            title=title, source=match.source, start=page, pages=length, notes=entry.notes
        )
        return SuccessResult(
            title=title,
            page=entry_page,
            source_path=match.source,
            source_page=page,
            matched_title=matched_title,
            score=score,
        )
    composer.start_section(title)
    first_page: int | None = None
    first_source: object = unique[0][3].source
    for _requested, _step, label, match in unique:
        page = entry.page if entry.page is not None else match.page
        length = entry.length if entry.length is not None else match.length
        entry_page = composer.add(
            title=label,
            source=match.source,
            start=page,
            pages=length,
            notes=entry.notes,
        )
        if first_page is None:
            first_page = entry_page
    composer.end_section()
    return SuccessResult(
        title=title,
        page=first_page or 0,
        source_path=first_source,  # type: ignore[arg-type]
        source_page=None,
        matched_title=matched_title,
        score=score,
    )


def process_items(
    items: list[SectionEntry | FileEntry],
    default_dir: Path,
    composer: Composer,
    libraries: LibraryCollection,
    key: str,
    show_key: bool,
    part: str | list[str] = "Concert",
    verbose: int = 0,
) -> None:
    """Process layout items recursively for one key edition."""
    for item in items:
        if isinstance(item, SectionEntry):
            composer.start_section(item.section)
            process_items(item.body, default_dir, composer, libraries, key, show_key, part, verbose)
            composer.end_section()
        else:
            result = process_file_entry(
                item,
                default_dir,
                composer,
                libraries,
                key,
                part,
                verbose,
            )
            if show_key:
                print(f"[{key}] {result.format()}")
            else:
                print(result.format())


def render(
    layout_path: Path,
    output: Path,
    keys: list[str] | None = None,
    base: Path | None = None,
    part: str | list[str] | None = None,
    verbose: int = 0,
) -> None:
    """Combine PDFs according to the YAML layout file.

    Renders one setbook per key edition and per --part option. `--part`
    selects instrument parts with substitution chains; other than Concert it
    writes `<stem>.<Part>.pdf`, combined values `<stem>.<A>+<B>.pdf`.
    """
    try:
        with open(str(layout_path), "r") as file:
            file_content = file.read()
            raw_data = yaml.safe_load(file_content)
    except yaml.YAMLError as e:
        # Add file path and line information to YAML parsing errors
        error_msg = f"YAML parsing error in {layout_path}"
        # Some YAML errors have problem_mark with line/column info
        problem_mark = getattr(e, "problem_mark", None)
        if problem_mark is not None:
            error_msg += f" at line {problem_mark.line + 1}, column {problem_mark.column + 1}"
        error_msg += f": {e}"
        raise ValueError(error_msg) from e

    # Parse and validate the YAML records using Pydantic
    try:
        layout = Layout.model_validate(raw_data)
    except ValidationError as e:
        # Use Pydantic's built-in error formatting
        error_msg = f"Validation error in {layout_path}:\n{e}"
        raise ValueError(error_msg) from e

    key_list = expand_keys(list(keys) if keys else [])

    base_dir, limit = resolve_base_dir(layout_path, base)
    libraries = load_project_libraries(base_dir, limit)

    layout_base = lexical_absolute(layout_path.parent)
    layout_entries: list[SectionEntry | FileEntry] = []
    for record in layout.root:
        if isinstance(record, ConfigEntry):
            for library_record in record.config:
                libraries.add(library_record.load(layout_base))
        else:
            layout_entries.append(record)

    default_dir = lexical_absolute(layout_path.parent)
    show_key = len(key_list) > 1
    part_specs = expand_part_specs(part)

    for label, subparts in part_specs:
        part_output = output_path_for_label(output, label if label != "Concert" else None)
        for key in key_list:
            key_output = output_path_for_key(part_output, key)
            with Composer(key_output, autosave=False) as composer:
                process_items(
                    layout_entries, default_dir, composer, libraries, key, show_key, subparts, verbose
                )
                composer.save()


def lookup_and_extract(
    title: str,
    output: Path | None,
    base: Path | None = None,
) -> None:
    """
    Look up a title in configured libraries and extract it to PDF.

    Args:
        title: The title to search for.
        output: Output path - if file, use as-is; if directory, save <title>.pdf there;
                if None, save <title>.pdf in current directory.
        base: Explicit config search base (`--base`).
    """
    base_dir, limit = resolve_base_dir(None, base)
    libraries = load_project_libraries(base_dir, limit)
    library_match = libraries.lookup_title(title)
    if library_match is None:
        print(f'No matches found for "{title}"')
        return

    matched_title = library_match.title
    output_path = (
        Path(f"{matched_title}.pdf")
        if output is None
        else output / f"{matched_title}.pdf"
        if output.is_dir()
        else output
    )
    location = format_source_location(library_match.source, library_match.page)
    print(f'Found "{matched_title}" in {location}')
    with Composer(output_path, autosave=False) as composer:
        composer.add(
            matched_title,
            library_match.source,
            start=library_match.page,
            pages=library_match.length,
        )
        composer.save()
    print(f"Extracted to: {output_path}")
