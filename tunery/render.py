"""PDF rendering logic for combining PDFs with table of contents."""

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PositiveInt,
    RootModel,
    ValidationError,
    field_validator,
)

from tunery.composer import Composer
from tunery.library import (
    LibraryCollection,
    LibraryMatch,
    LibraryRecord,
    PartHit,
    ScoreFolder,
    load_project_libraries,
)
from tunery.parts import canonical_part_name, normalize_text, parse_part_name
from tunery.viewpath import ViewPath


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
    # Requested part -> part this title uses for it (`part:` or `parts:`).
    part: dict[str, str] | None = Field(
        default=None, validation_alias=AliasChoices("part", "parts")
    )
    # Preferred variation or free tags (`tag:` or `tags:`, one or a list).
    tag: list[str] | None = Field(
        default=None, validation_alias=AliasChoices("tag", "tags")
    )
    # Preferred library-relative path (`path:` or `paths:`, one or a list).
    path: list[str] | None = Field(
        default=None, validation_alias=AliasChoices("path", "paths")
    )

    @field_validator("tag", "path", mode="before")
    @classmethod
    def _one_or_list(cls, value: object) -> object:
        return [value] if isinstance(value, str) else value

    def model_post_init(self, context: object, /) -> None:
        if not self.file and not self.title:
            raise ValueError("Either 'file' or 'title' must be provided")


class SectionEntry(BaseModel):
    """A section containing nested file entries or subsections."""

    section: str
    body: list["FileEntry | SectionEntry"] = Field(default_factory=list)


class NamedParts(BaseModel):
    """Named part selections in a layout config (`part:` or `parts:`)."""

    part: dict[str, str] = Field(validation_alias=AliasChoices("part", "parts"))


class ConfigEntry(BaseModel):
    """Configuration entry adding libraries and named parts."""

    config: list[LibraryRecord | NamedParts]


class Layout(RootModel[list[ConfigEntry | SectionEntry | FileEntry]]):
    """The complete layout schema - a list of config, sections, and/or file entries."""



# Column names (ignoring case and surrounding spaces) that hold titles in a .csv
# input, in order of preference.
CSV_TITLE_COLUMNS = ("Title",)

RHYTHM_SECTION = ["Piano", "Guitar", "Bass", "Drums"]
# Part bookmarks under a title, for readers that show only one bookmark level.
PART_BOOKMARK_PREFIX = "_ "
RHYTHM_SECTION_NAMES = {"rs": "RS", "rhythm": "Rhythm", "rhythm section": "Rhythm Section"}


def output_path_for_label(output: Path, label: str | None) -> Path:
    """Derive output path from a part-spec label (RS stays .RS.pdf)."""
    if label is None or label in ("Concert", "C"):
        return output
    # Don't double-suffix when -o already ends with the part or an alias.
    if _same_part(output.stem.split(".")[-1], label):
        return output
    return output.with_name(f"{output.stem}.{label}{output.suffix}")


def _same_part(a: str, b: str) -> bool:
    """Whether two part names (aliases included) name the same part."""
    if normalize_text(a) in RHYTHM_SECTION_NAMES and normalize_text(b) in RHYTHM_SECTION_NAMES:
        return True
    return canonical_part_name(a) == canonical_part_name(b)


def _expand_part(raw: str, named: dict[str, str]) -> tuple[str, list[str]]:
    """One --part element as (label, parts): a name from the layout config,
    the rhythm section, or a single part."""
    norm = normalize_text(raw)
    if norm in named:
        members: list[str] = []
        for sub in named[norm].split(","):
            if sub.strip():
                members.extend(_expand_part(sub.strip(), {})[1])
        return raw.strip(), list(dict.fromkeys(members))
    if norm in RHYTHM_SECTION_NAMES:
        return RHYTHM_SECTION_NAMES[norm], RHYTHM_SECTION
    canon = parse_part_name(raw)
    return canon, [canon]


def expand_part_specs(
    parts: list[str] | str | None, named: dict[str, str] | None = None
) -> list[tuple[str, list[str]]]:
    """Normalize --part options into (output label, subparts) per output.

    Each option is one output. A comma-separated value combines its parts
    under a '-'-joined label. Names defined in the layout config and the
    rhythm section (RS/Rhythm/Rhythm Section) expand to their parts.
    """
    if parts is None:
        return [("Concert", ["Concert"])]
    if isinstance(parts, str):
        parts = [parts]
    named = {normalize_text(name): value for name, value in (named or {}).items()}
    specs: list[tuple[str, list[str]]] = []
    for spec in parts:
        expanded = [_expand_part(s.strip(), named) for s in spec.split(",") if s.strip()]
        if not expanded:
            specs.append(("Concert", ["Concert"]))
            continue
        subparts = list(dict.fromkeys(p for _, members in expanded for p in members))
        specs.append(("-".join(label for label, _ in expanded), subparts))
    return specs or [("Concert", ["Concert"])]


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


def display_title(explicit: str | None, resolved: Path) -> str:
    """Display title for an entry."""
    if explicit:
        return explicit
    return resolved.stem


def process_file_entry(
    entry: FileEntry,
    default_dir: Path,
    composer: Composer,
    libraries: LibraryCollection | None = None,
    part: str | list[str] = "Concert",
    verbose: int = 0,
    part_prefix: str = "",
    book: str | None = None,
    named: dict[str, str] | None = None,
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
    if entry.file:
        folder = ViewPath(default_dir, entry.file).absolute()
        if folder.is_dir():
            return select_and_add(
                entry.title or folder.name,
                entry,
                LibraryCollection([ScoreFolder(folder)]),
                subparts,
                composer,
                verbose,
                part_prefix,
                book,
                named,
            )
        if folder.suffix.lower() != ".pdf" and not folder.exists():
            raise ValueError(f"Score folder not found: {folder}")
        library_match = libraries.lookup_file(entry.file) if libraries else None
        if library_match is not None:
            input_pdf_path = library_match.source
        else:
            input_pdf_path = ViewPath(default_dir, entry.file).absolute()
        title = display_title(entry.title, input_pdf_path)
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
            source_path=library_match.shown if library_match else input_pdf_path,
            source_page=page,
            matched_title=matched_title,
            score=score,
        )
    if not entry.title:
        raise ValueError("Entry must have either 'file' or 'title'")
    title = entry.title
    if libraries is None:
        return NotFoundResult(title=title)
    return select_and_add(
        title, entry, libraries, subparts, composer, verbose, part_prefix, book, named
    )


def _add_hit(
    composer: Composer, bookmark_title: str, match: LibraryMatch, entry: FileEntry
) -> int:
    """Add one hit's ranges in list order under a single bookmark: the first
    range takes `bookmark_title`, continuation ranges take none."""
    if entry.page is None and entry.length is None and match.ranges:
        ranges = list(match.ranges)
    else:
        ranges = [(entry.page if entry.page is not None else match.page,
                   entry.length if entry.length is not None else match.length)]
    first_page: int | None = None
    for i, (start, pages) in enumerate(ranges):
        entry_page = composer.add(
            title=bookmark_title,
            source=match.source,
            start=start,
            pages=pages,
            notes=entry.notes,
            bookmark=(i == 0),
        )
        if first_page is None:
            first_page = entry_page
    assert first_page is not None
    return first_page


def select_and_add(
    title: str,
    entry: FileEntry,
    libraries: LibraryCollection,
    subparts: list[str],
    composer: Composer,
    verbose: int,
    part_prefix: str,
    book: str | None = None,
    named: dict[str, str] | None = None,
) -> ProcessEntryResult:
    """Choose the title's charts for the requested parts and add them: one
    chart flat under the title, several nested under a title bookmark."""
    if entry.part:
        names = {normalize_text(name) for name in (named or {})}
        this_book = normalize_text(book) if book else None
        part_map: dict[str, str] = {}
        replaced = False
        for asked, used in entry.part.items():
            if normalize_text(asked) in names:
                if normalize_text(asked) == this_book:
                    subparts = [parse_part_name(used, title)]
                    replaced = True
                    break
                continue
            part_map[parse_part_name(asked, title)] = parse_part_name(used, title)
        if not replaced:
            subparts = list(dict.fromkeys(part_map.get(p, p) for p in subparts))
    selection = libraries.select_parts(
        title, subparts, entry.key, entry.tag, entry.path
    )
    if selection is None:
        return NotFoundResult(title=title)
    for warning in selection.warnings:
        print(warning)
    for sub in selection.missing:
        print(f'missing {sub} for "{title}"')
    substituted: dict[str, list[str]] = {}
    for hit in selection.hits:
        if hit.step != hit.requested:
            substituted.setdefault(hit.step, []).append(hit.requested)
    if substituted:
        groups = [
            f"{', '.join(reqs)} → {step}" for step, reqs in substituted.items()
        ]
        print(f"{title}: {'; '.join(groups)}")
    if verbose >= 1:
        for hit in selection.hits:
            print(f'  "{title}" {hit.requested}: {hit.match.source.name}')
    if verbose >= 2:
        for lib_desc, path, cand in libraries.all_candidates(title):
            if cand is None:
                print(f'  candidate "{title}" in "{lib_desc}" ({path.name}): indexed')
            else:
                print(
                    f'  candidate "{title}" in "{lib_desc}" ({path.name}): '
                    f"part={cand.part}, key={cand.score_key}, "
                    f"var={cand.variations}, free={cand.free_tags}, "
                    f"noise={cand.noise}, rules={cand.rules_fired}"
                )
    if not selection.hits:
        return NotFoundResult(title=title)
    # One chart per file (and page range), in requested-part order.
    first_hits: dict[tuple[Path, int | None], PartHit] = {}
    for hit in selection.hits:
        first_hits.setdefault((hit.match.source, hit.match.page), hit)
    unique = list(first_hits.values())
    if len(unique) == 1:
        match = unique[0].match
        entry_page = _add_hit(composer, title, match, entry)
        return SuccessResult(
            title=title,
            page=entry_page,
            source_path=match.shown,
            source_page=entry.page if entry.page is not None else match.page,
        )
    composer.start_section(title)
    first_page: int | None = None
    for hit in unique:
        entry_page = _add_hit(
            composer, f"{part_prefix}{hit.label}", hit.match, entry
        )
        if first_page is None:
            first_page = entry_page
    composer.end_section()
    return SuccessResult(
        title=title,
        page=first_page or 0,
        source_path=unique[0].match.shown,
        source_page=None,
    )


def process_items(
    items: list[SectionEntry | FileEntry],
    default_dir: Path,
    composer: Composer,
    libraries: LibraryCollection,
    part: str | list[str] = "Concert",
    verbose: int = 0,
    part_prefix: str = "",
    book: str | None = None,
    named: dict[str, str] | None = None,
) -> None:
    """Process layout items recursively for one setbook."""
    for item in items:
        if isinstance(item, SectionEntry):
            composer.start_section(item.section)
            process_items(
                item.body,
                default_dir,
                composer,
                libraries,
                part,
                verbose,
                part_prefix,
                book,
                named,
            )
            composer.end_section()
        else:
            result = process_file_entry(
                item,
                default_dir,
                composer,
                libraries,
                part,
                verbose,
                part_prefix,
                book,
                named,
            )
            print(result.format())


def _layout_titles(items: list[SectionEntry | FileEntry]) -> list[str]:
    """Every chart title in a layout, in order, sections included."""
    titles: list[str] = []
    for item in items:
        if isinstance(item, SectionEntry):
            titles.extend(_layout_titles(item.body))
        elif item.title is not None:
            titles.append(item.title)
    return titles


def _filter_titles(
    items: list[SectionEntry | FileEntry], wanted: set[str]
) -> list[SectionEntry | FileEntry]:
    """Layout items reduced to the requested titles."""
    kept: list[SectionEntry | FileEntry] = []
    for item in items:
        if isinstance(item, SectionEntry):
            kept.append(
                SectionEntry(
                    section=item.section, body=_filter_titles(item.body, wanted)
                )
            )
        elif item.title in wanted:
            kept.append(item)
    return kept


def read_layout_records(layout_path: Path) -> object:
    """Layout records of a YAML layout, or of a title list: a .txt file with
    one title per line, or a .csv file with a title column."""
    suffix = layout_path.suffix.lower()
    if suffix in (".txt", ".csv"):
        titles = _read_titles(layout_path, csv_input=suffix == ".csv")
        if not titles:
            raise ValueError(f"No titles in {layout_path}")
        return [{"title": title} for title in titles]
    try:
        with open(str(layout_path), "r") as file:
            file_content = file.read()
            return yaml.safe_load(file_content)
    except yaml.YAMLError as e:
        # Add file path and line information to YAML parsing errors
        error_msg = f"YAML parsing error in {layout_path}"
        # Some YAML errors have problem_mark with line/column info
        problem_mark = getattr(e, "problem_mark", None)
        if problem_mark is not None:
            error_msg += f" at line {problem_mark.line + 1}, column {problem_mark.column + 1}"
        error_msg += f": {e}"
        raise ValueError(error_msg) from e


def _read_titles(path: Path, *, csv_input: bool) -> list[str]:
    """Non-empty trimmed titles: one per line, or from the title column."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"Cannot read {path}: it is not UTF-8 text") from e
    if not csv_input:
        lines = text.splitlines()
    else:
        reader = csv.DictReader(text.splitlines())
        header = {(name or "").strip().casefold(): name for name in reader.fieldnames or []}
        column = next(
            (header[wanted.casefold()] for wanted in CSV_TITLE_COLUMNS if wanted.casefold() in header),
            None,
        )
        if column is None:
            found = ", ".join(name for name in reader.fieldnames or []) or "none"
            raise ValueError(
                f"No title column in {path}: expected {' or '.join(CSV_TITLE_COLUMNS)} "
                f"(found {found})"
            )
        lines = [row[column] or "" for row in reader]
    return [line.strip() for line in lines if line.strip()]


def render(
    layout_path: Path,
    output: Path,
    base: Path | None = None,
    part: str | list[str] | None = None,
    verbose: int = 0,
    explicit_bookmark_hierarchy: bool = False,
    test: list[str] | None = None,
) -> None:
    """Combine PDFs according to the YAML layout file.

    Relative `layout_path` and `output` join `base` (default: the logical
    working directory); configuration is found from the layout's folder.

    Renders one setbook per --part option. `--part`
    selects instrument parts with substitution chains; other than Concert it
    writes `<stem>.<Part>.pdf`, combined values `<stem>.<A>-<B>.pdf`.
    """
    layout_path = ViewPath(layout_path, base=base).absolute()
    output = ViewPath(output, base=layout_path.base).absolute()
    raw_data = read_layout_records(layout_path)

    # Parse and validate the YAML records using Pydantic
    try:
        layout = Layout.model_validate(raw_data)
    except ValidationError as e:
        # Use Pydantic's built-in error formatting
        error_msg = f"Validation error in {layout_path}:\n{e}"
        raise ValueError(error_msg) from e

    libraries = load_project_libraries(layout_path.parent)
    if verbose >= 1:
        print(f"configuration {libraries.config_path or 'not found'}")

    layout_base = layout_path.parent
    layout_entries: list[SectionEntry | FileEntry] = []
    named_parts: dict[str, str] = {}
    for record in layout.root:
        if isinstance(record, ConfigEntry):
            for config_record in record.config:
                if isinstance(config_record, NamedParts):
                    named_parts.update(config_record.part)
                else:
                    libraries.add(config_record.load(layout_base))
        else:
            layout_entries.append(record)

    default_dir = layout_path.parent
    part_specs = expand_part_specs(part, named_parts)

    wanted: set[str] | None = None
    if test is not None:
        names = [name.strip() for name in test]
        if not any(name in (".", "") for name in names):
            wanted = set(names)
            unknown = [
                name for name in wanted if name not in _layout_titles(layout_entries)
            ]
            if unknown:
                raise ValueError(
                    f"Title not in layout: {', '.join(sorted(unknown))}"
                )
        if wanted is not None:
            layout_entries = _filter_titles(layout_entries, wanted)

    for label, subparts in part_specs:
        if test is not None:
            # Dry run: -vv results to a scratch PDF that is deleted.
            with (
                tempfile.TemporaryDirectory() as scratch,
                Composer(Path(scratch) / "dry.pdf", autosave=False) as composer,
            ):
                process_items(
                    layout_entries,
                    default_dir,
                    composer,
                    libraries,
                    subparts,
                    2,
                    PART_BOOKMARK_PREFIX if explicit_bookmark_hierarchy else "",
                    label,
                    named_parts,
                )
            continue
        part_output = output_path_for_label(output, label if label != "Concert" else None)
        with Composer(part_output, autosave=False) as composer:
            process_items(
                layout_entries,
                default_dir,
                composer,
                libraries,
                subparts,
                verbose,
                PART_BOOKMARK_PREFIX if explicit_bookmark_hierarchy else "",
                label,
                named_parts,
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
        base: Root of the file-system view (`--base`); configuration is
              found from it, and a relative `output` joins it.
    """
    root = ViewPath(base or ".").absolute()
    libraries = load_project_libraries(root)
    library_match = libraries.lookup_title(title)
    if library_match is None:
        print(f'No matches found for "{title}"')
        return

    matched_title = library_match.title
    output = ViewPath(output or ".", base=root).absolute()
    output_path = output / f"{matched_title}.pdf" if output.is_dir() else output
    location = format_source_location(library_match.shown, library_match.page)
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
