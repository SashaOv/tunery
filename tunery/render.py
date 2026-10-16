"""PDF rendering logic for combining PDFs with table of contents."""

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
    LibraryRecord,
    PartHit,
    ScoreFolder,
    lexical_absolute,
    load_project_libraries,
    logical_cwd,
)
from tunery.parts import canonical_part_name, normalize_text, parse_part_name


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

    @field_validator("tag", mode="before")
    @classmethod
    def _tag_list(cls, value: object) -> object:
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
    under a '+'-joined label. Names defined in the layout config and the
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
        specs.append(("+".join(label for label, _ in expanded), subparts))
    return specs or [("Concert", ["Concert"])]


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """
    Resolve a file path relative to base_dir if it's relative, otherwise return as-is.

    Normalizes lexically without dereferencing intermediate symlinks, so `..`
    climbs above symlinked parents instead of their targets.
    """
    path = Path(path_str)
    return lexical_absolute(path, base_dir)


def view_base(base: Path | None) -> Path:
    """Root of the file-system view: `--base`, else the logical working
    directory. Relative command-line paths join it."""
    return lexical_absolute(base) if base is not None else logical_cwd()


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
        folder = resolve_path(entry.file, default_dir)
        if folder.is_dir():
            return select_and_add(
                entry.title or folder.name,
                entry,
                LibraryCollection([ScoreFolder(folder)]),
                subparts,
                composer,
                verbose,
                part_prefix,
            )
        if folder.suffix.lower() != ".pdf" and not folder.exists():
            raise ValueError(f"Score folder not found: {folder}")
        library_match = libraries.lookup_file(entry.file) if libraries else None
        if library_match is not None:
            input_pdf_path = library_match.source
        else:
            input_pdf_path = resolve_path(entry.file, default_dir)
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
    return select_and_add(title, entry, libraries, subparts, composer, verbose, part_prefix)


def select_and_add(
    title: str,
    entry: FileEntry,
    libraries: LibraryCollection,
    subparts: list[str],
    composer: Composer,
    verbose: int,
    part_prefix: str,
) -> ProcessEntryResult:
    """Choose the title's charts for the requested parts and add them: one
    chart flat under the title, several nested under a title bookmark."""
    if entry.part:
        mapping = {
            parse_part_name(asked, title): parse_part_name(used, title)
            for asked, used in entry.part.items()
        }
        subparts = list(dict.fromkeys(mapping.get(p, p) for p in subparts))
    selection = libraries.select_parts(title, subparts, entry.key, entry.tag)
    if selection is None:
        return NotFoundResult(title=title)
    for warning in selection.warnings:
        print(warning)
    for sub in selection.missing:
        print(f'missing {sub} for "{title}"')
    if verbose >= 1:
        for hit in selection.hits:
            print(f'  "{title}" {hit.requested}: {hit.match.source.name}')
            if hit.step != hit.requested:
                print(f"  {hit.requested} → {hit.step}")
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
        page = entry.page if entry.page is not None else match.page
        length = entry.length if entry.length is not None else match.length
        entry_page = composer.add(
            title=title, source=match.source, start=page, pages=length, notes=entry.notes
        )
        return SuccessResult(
            title=title,
            page=entry_page,
            source_path=match.shown,
            source_page=page,
        )
    composer.start_section(title)
    first_page: int | None = None
    for hit in unique:
        page = entry.page if entry.page is not None else hit.match.page
        length = entry.length if entry.length is not None else hit.match.length
        entry_page = composer.add(
            title=f"{part_prefix}{hit.label}",
            source=hit.match.source,
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
            )
            print(result.format())


def render(
    layout_path: Path,
    output: Path,
    base: Path | None = None,
    part: str | list[str] | None = None,
    verbose: int = 0,
    explicit_bookmark_hierarchy: bool = False,
) -> None:
    """Combine PDFs according to the YAML layout file.

    Relative `layout_path` and `output` join `base` (default: the logical
    working directory); configuration is found from the layout's folder.

    Renders one setbook per --part option. `--part`
    selects instrument parts with substitution chains; other than Concert it
    writes `<stem>.<Part>.pdf`, combined values `<stem>.<A>+<B>.pdf`.
    """
    root = view_base(base)
    layout_path = lexical_absolute(layout_path, root)
    output = lexical_absolute(output, root)
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

    for label, subparts in part_specs:
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
    root = view_base(base)
    libraries = load_project_libraries(root)
    library_match = libraries.lookup_title(title)
    if library_match is None:
        print(f'No matches found for "{title}"')
        return

    matched_title = library_match.title
    output = lexical_absolute(output or Path(), root)
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
