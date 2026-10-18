# Tunery

**Goal:** Build a printable PDF setbook by stitching together pages from
existing sheet-music PDFs.

## Concepts

### Layout

A layout is an ordered YAML description of charts and nested sections in an
output setbook. A chart can name a PDF file, a folder of PDFs, or a tune
title.

### Library

A Library is a searchable collection of tunes. A successful lookup returns the
source PDF, canonical title, starting page, and page count.

Tunery supports:

- **Directory libraries:** folders of PDFs whose file and folder names give
  each item's title, part and variations.
- **Indexed PDF libraries:** a source PDF paired by its library record with a
  JSON mapping from tune titles to page ranges.
- **Library collections:** ordered directory and indexed PDF libraries searched
  as one collection, optionally loaded from a YAML collection file.

Each library uses either normalized exact matching or exact-then-fuzzy
matching.

### Part

What one player reads, in order of specificity:

- instrument + role (`Alto Sax 2`, `Horn 1`; a role may be played by
  different instruments);
- instrument (`Piano`, `Tenor Sax`, `Vocal`);
- group (`Concert`, `Bb`, `Eb`, `Bass Clef`, `Rhythm`);
- kind (`Lead Sheet`, `Score`, `Lyrics`, `Solo`, `Changes`).

A **variation** distinguishes two items of the same title and part: score key
(`Concert F`, `G Major`), version (`v2`) and free tags (`JBM Arrangement`,
an arranger's name). Variations are ignored unless a chart selects them.

### Library type

`combo` (default) or `big band`. The type sets the naming rules (see Name
interpretation) and the default part of an untagged item: the Concert lead
sheet for combo, a jumbo file (see Score formats) for big band. A conductor
score (`conductor`, `score`, `condensed score`) is a part of its own
(`Conductor`) in both: listed, never substituted for another part.

### File-system view

- The **base** directory is `--base` if given, else the logical working
  directory (`$PWD`). Relative command-line paths are joined to it.
- Paths are navigated lexically: `..` is collapsed and directory symlinks are
  never resolved. A directory symlink (`GDrive`, `Vault`) is a **link**: going
  down through it enters the linked tree; going up from inside it returns
  through the link, never to the linked tree's physical parents.
- Messages show paths in this view (`GDrive/SheetMusic/…`), not resolved ones.

## Deployable

`tunery` is both a CLI and a Python package.

### CLI

- **render** `LAYOUT_YAML [-o OUTPUT_PDF]` builds a setbook. The default output
  is `<layout_basename>.pdf` next to the layout.
  - `--part PART` selects the part (default `Concert`). Each `--part` is its
    own output; a comma-separated value combines parts in one output.
  - `--test TITLE` (repeatable) resolves the named titles, prints `-vv`-level
    results and writes no PDF. `.` or `""` means every title. A title not in
    the layout is an error.
  - `-v`/`--verbose 1` and `-vv`/`--verbose 2` add detail (see Messages).
  - `--explicit-bookmark-hierarchy` prefixes part bookmarks with `_ ` (`_
    Bass`), for readers that show one bookmark level.
- **lookup** `TITLE... [-o OUTPUT]` extracts requested titles. If `OUTPUT` is a
  directory, Tunery writes `<canonical_title>.pdf` inside it. If it is a file,
  Tunery uses it as-is. The default is
  `<canonical_title>.pdf` in the current directory.
- Both commands take `--base DIR` (see File-system view).

### Python API

- Import `Composer`, `DirectoryLibrary`, `IndexedPdfLibrary`, `Library`,
  and `LibraryCollection` from the package root.
- Construct `Composer(output_path)` to build a setbook programmatically.
- Call `add(title, source, start=None, pages=None)` to append pages from a PDF.
- Call `start_section(title)` and `end_section()` to create nested outline
  sections.
- Save the output after additions. Use `close()` or a context manager to
  release the PDF.

## Configuration / Inputs

### Project configuration

The project configuration file is named `tunery.yaml` or `.tunery.yaml`. It
contains an ordered list of library records:

```yaml
- library: indexes/real-book.json
  source: books/Real Book.pdf
  shift: 0
  match: fuzzy
- library: handouts
  match: exact
```

Fields:

- **library** (required): a directory, tune-index JSON file, or YAML library
  collection file.
- **source** (required for indexed PDF libraries): source PDF path. Directory
  libraries and YAML collections must not specify `source`.
- **shift** (optional): integer added to every indexed tune page; defaults to
  `0`. Directory libraries and YAML collections must not specify `shift`.
- **match** (optional): `exact` or `fuzzy`; defaults to `exact`.
- **type** (optional): `combo` or `big band`; defaults to `combo`.
- **index** (optional, directory libraries): the part index (below).

Resolve both `library` and `source` paths relative to the file containing the
library record: `tunery.yaml` for project libraries or the layout file for
layout libraries.

A YAML library collection file contains an ordered list of library records
using the same schema, for example `/path/to/rb-index/tunery.yaml`. Referencing
it includes all of its libraries in order at that position:

```yaml
- library: /path/to/rb-index/tunery.yaml
```

Resolve paths inside the collection relative to the collection file in the
file-system view; a collection file that is itself a symlink resolves against
its real location. Collections may nest; circular includes are an error. A
`match` or `type` set on the collection reference applies only to inner
records that do not set it explicitly; an inner value always wins.

#### Part index

A directory library's `index:` lists where parts sit inside jumbo files:

```yaml
- library: ./
  type: big band
  match: fuzzy
  index:
    - title: Bandstand Boogie
      parts: {bass: 15, piano: [17, 3]}
    - title: Come Dance With Me
      parts: {bass: [[26, 1], [25, 1]]}
    - title: I've Got You Under My Skin
      file: I've Got You Under My Skin - Ebm.pdf
      parts: {bass: [27, 2]}
```

- An entry's title is matched like a layout title (fuzzily in fuzzy
  libraries). Its file is the title's jumbo file, or `file:` (relative to the
  library) when the title finds several jumbo files or the file's name reads
  as something else.
- A part value is `page`, `[page, length]`, or a list of ranges rendered in
  list order. Part names must be known parts.
- Each indexed part is a candidate of the title at that part's own step.
- An entry whose title finds no jumbo file, or several without `file:`, is
  warned, and its parts count as missing.

### Indexed PDF library

An indexed PDF library JSON file is an array of tune records:

```json
[
  {"title": "Autumn Leaves", "page": 39},
  {"title": "Song B", "page": 20, "pages": 2}
]
```

Fields:

- **title** (required): canonical tune title.
- **page** (required): positive starting page before applying the library
  record's `shift`.
- **pages** (optional): positive page count; defaults to `1`.

Tunery reads this array directly and pairs it with the `source` and optional
`shift` from the library record.

### Layout YAML

A layout is a list containing config, chart, and section records.

A config record adds libraries after the project libraries, and may name
part selections:

```yaml
- config:
    - library: ../../handouts
      match: exact
    - library: ../../indexes/real-book.json
      source: ../../books/Real Book.pdf
      match: fuzzy
    - part:
        full: Concert, Trumpet, Tenor Sax, Rhythm
        norm: Tenor Sax
```

Resolve layout `library` and `source` paths relative to the layout file. A
`part:` (or `parts:`) item defines names usable as `--part` values;
`--part norm` writes `<stem>.norm.pdf`.

A chart record names a file, a title, or both:

```yaml
- file: Songbook.pdf
  page: 12
  length: 2
  title: Latin Medley
  notes: "Watch ending — fermata on bar 32"
- title: Norwegian Wood
  key: F
  part: {Tenor Sax: Horn 1, Trumpet: Horn 2}
- title: Funk in Deep Freeze
  parts: {norm: Flute}
- title: I Love Lucy
  tag: v2
  path: Vocals
```

Chart fields:

- **file** (optional): PDF path or filename, or a folder whose PDFs are this
  title's charts (chosen by part, without matching their names to the
  title). Either `file` or `title` is required.
- **title** (optional): display title and title lookup key.
- **page** (optional): positive starting page.
- **length** (optional): positive page count.
- **notes** (optional): text drawn at the bottom of every included page.
- **part** / **parts** (optional): maps requested parts to the parts this
  title uses for them. A key that is a part applies in every book; a key that
  names a layout part selection replaces the title's parts in that book only.
  Which instrument reads `Horn 1` is decided only here.
- **key** (optional): score key; keeps only candidates in that key (a key
  without a mode matches any mode).
- **tag** / **tags** (optional): one or several variation or free tags to
  prefer.
- **path** / **paths** (optional): one or several fragments of the path
  inside the library to prefer, ignoring case. A value with `*` or `?` is a
  glob on the whole path (`*` crosses folders).

A section creates a nested PDF outline:

```yaml
- section: Set 1
  body:
    - title: Country
    - file: Groove.pdf
      title: Groove Standard
    - section: Medley
      body:
        - title: Song A
        - title: Song B
```

Sections can nest to any depth and may be empty.

## Behavior

### Configuration discovery

Search for the project configuration from a starting directory and walk up
to the filesystem root, in the file-system view. The starting directory is
the layout's directory for `render` and the base for `lookup`; the search is
otherwise the same. In each directory look for
`tunery.yaml` and `.tunery.yaml`; the first directory holding one wins, and
holding both is an error. Do not merge project configurations.

If the found configuration is a symbolic link, dereference it and resolve
relative `library` and `source` paths against the real file's directory.

If no project configuration exists, print
`Project configuration was not found, using defaults` and use the starting
directory as a directory library.

### Library discovery

- A directory library holds every PDF below it, recursively, following
  symlinks without revisiting a real directory.
- `_archive` folders are never scanned. Other folders whose names start with
  `_` (`_Petes`) are scanned, but their names are ignored.
- An item's **name texts** are its stem and its innermost title folder.

### Name interpretation

- **Normalization:** lowercase, `’`→`'`, apostrophes dropped, punctuation and
  `_` → space, compared with spaces removed (`SoWhat` = `So What`). Stem
  cleanup unwraps `Finale 2004a - [<name>]`, drops a leading `Copy of ` (kept
  as a noise word) and a trailing `(1)`.
- **Boundaries:** hard boundaries are ` - `, `-`, `.`, `,`, `( )`, `[ ]`,
  `:`; a space is soft. A title may end at a soft boundary only when
  everything after it is tags (`Morning Piano` is a Morning part,
  `Morning Dance` is not).
- **Title match:** a layout title matches a name text that starts with it at
  a boundary: exact first, then, in `fuzzy` libraries, by comparing the
  title with each word prefix of the name text (ratio ≥ 0.9). What follows
  the title is read as tags. When only the folder matches, the stem's tags
  are read right to left up to the first word that isn't a tag.
- **Shortened title:** when no name text matches, a file name that starts
  with the title and continues with more words is that title's untagged
  chart, if it is the only such name; the match is warned (`matched
  "<title>" to "<name>"`). Two or more such names are warned (`several names
  start with "<title>": …`) and the title is not found. A title with its own
  candidates never takes a longer name.
- **Rules:** tags are read right to left. At each position the longest span
  wins, then rule priority. Priorities per library type live in `RULES` in
  `tunery/parts.py` (0 disables a rule):

  | Rule | Matches | Combo | Big band |
  |---|---|---|---|
  | Explicit score key | `Concert <key>`, `in <key>`, `<key> major/minor`, `<key>m` | 1 | 1 |
  | Instrument | vocabulary below, with roles (`Horn 1`, `Alto Sax 2`) | 1 | 1 |
  | Group | `Concert`, `C Instruments`, `Melody`, `Rhythm`, `Bass Clef`; bare `Bb`/`Eb` in combo | 2 | 1 |
  | Bare score key | bare `Bb`, `Eb` (`Sweet Home Chicago.Bb`) | 0 | 1 |
  | Conductor | `conductor`, `score`, `condensed score` | 0 | 1 |
  | Kind | `Lead Sheet`, `Score`, `Lyrics`, `Solo`, `Changes`, `Structure`, `Harmony` | 1 | 1 |
  | Page split | `Pg N`, `P1`, `_pg 2` | 1 | 1 |
  | Noise | `original`, `rescan`, `org`, `iRealPro`, a leading `Copy of`, track numbers | 1 | 1 |
  | Filler | `part`, `in`, `with`, `the`, `a`, `n`, `and`, `&` (not counted) | 1 | 1 |
  | Version | `v2`, `v3`, …: a variation | 1 | 1 |
  | Jumbo default | an untagged file is a jumbo; a group file is a combined set | 0 | 1 |
  | Title initials | in a title folder, a name starting with the title's initials (three or more letters) is that title: `fmttmbass` → Bass; a number glued to Piano is a page (`fmttmpno1`) | 0 | 1 |

- **Vocabulary** (instrument → group): Soprano Sax, Tenor Sax, Clarinet, Bass
  Clarinet, Trumpet, Flugelhorn → Bb; Alto Sax, Bari Sax → Eb; Trombone, Bass
  Trombone → Bass Clef; Flute, Vocal → Concert; Piano, Guitar, Bass, Drums,
  Percussion → Rhythm; Saxophone (unknown key) → no group. Aliases
  (`tenor`, `ts`, `bone`, `pno`, `keys`, `bss`, `voice`, …) live in
  `tunery/parts.py`. Bass is in Rhythm only: a Bass Clef chart never stands
  in for a bass part.
- **Dual scores:** two instruments joined in one segment (`Piano:Guitar`,
  `piano-guitar`, `pnoguit`) are one score with both parts. A multi-word
  instrument (`Bass Clarinet`) or an instrument in parentheses is not a
  dual.
- **Part of an item:** the rightmost instrument (with role), or both
  instruments of a dual score; else the group, then the kind, then the
  library type's default part. An earlier instrument is a variation
  (`All of Me (Vocal)-Piano` → Piano). A role in parentheses after an
  instrument is the role it plays (`tenor (horn 2)` is Tenor Sax as Horn 2).
- **Free tags:** words no rule takes after a hard boundary are free tags;
  nothing is guessed.
- **Score formats** (big band): a file holds one part, every part in
  sequence (**jumbo**, named for no part), or a few related parts (**combined
  set**, named for their group). A jumbo or combined set is never reached by
  substitution; its parts come from the part index.

### Lookup precedence and matching

Load project libraries first, followed by layout libraries in record order.
Search from last to first, so later libraries have higher priority.

For `title:` entries and the `lookup` command:

1. Try each library, highest priority (latest record) first.
2. Match titles as in Name interpretation; a `fuzzy` library tries fuzzy
   title matching after exact matching fails.
3. Return the highest-priority library with a candidate (see Part
   selection for `render`).

For `file:` entries, search directory libraries in the same priority order.
Indexed PDF libraries do not participate in file lookup. If no directory
library resolves the file, resolve it relative to the layout. Collapse `.`
and `..` lexically without dereferencing intermediate symlinks, so a layout
reached through a symlink (for example `GDrive` → a cloud folder) still
climbs to siblings such as `Vault`.

Indexed PDF lookup preserves the indexed page range and adds the library
record's `shift` to the starting page. The resulting page must be positive.

### Part selection

- **Substitution chains:** a requested part falls back step by step:
  - Instrument N → Instrument → group → Concert rank → Melody → Vocal.
  - Piano, Bass, Drums → Rhythm → Concert rank → Melody → Vocal.
  - Guitar → Piano → Rhythm → Concert rank → Melody → Vocal.
  - Concert → Concert rank → Melody → Piano/Guitar → Vocal.
  - The Concert rank is Concert or untagged, then Lead Sheet, then a Rhythm
    chart. A Piano/Guitar dual score matches Piano and Guitar directly.
- **Rhythm section:** `Rhythm`, `RS` and `Rhythm Section` select
  `Piano,Guitar,Bass,Drums`, also inside a comma list. Each member follows
  its own chain.
- **Ranks:** an instrument and its key group are one rank (Tenor Sax and Bb;
  Alto Sax, Bari Sax and Eb; Trombone and Bass Clef). Piano and Guitar are
  one rank.
- **Library:** the highest-priority library (the latest record) with a
  candidate on any step of any requested part's chain wins the title. All
  parts of a title come from that library; a part it lacks is missing, never filled from
  another library.
- **Sources:** each folder holding candidates for a title is a source. The
  top source has the most requested parts at their own step, then the fewest
  extra words, then comes first in path order.
- **Choice per part:** the best rank any source reaches; among its
  candidates, in order:
  1. those carrying the chart's `tag:`;
  2. those under the chart's `path:`;
  3. the fewest substitution steps (a `part:` mapping is not a
     substitution);
  4. those in the top source;
  5. the fewest extra words (variations, free tags and noise words);
  6. path order.
- **Key, tag and path misses:** when no candidate matches a chart's `key:`,
  `tag:` or `path:`, all candidates are used and the miss is warned (`no
  <key or tag> version of "<title>"`, `no candidate under "<path>" for "<title>"`).
- **Combined parts:** each requested part resolves through its chain. A file
  reached by two parts is included once. A part whose chain finds nothing
  for a found title is reported (`missing <part> for "<title>"`).
- **Indexed PDF libraries** answer any part request with their indexed
  pages, as concert sources.
- An unknown part name in `--part`, a part selection, a chart mapping or a
  part index is an error naming the value (and, for a chart, its title).

### Rendering and lookup

Rendering processes layout records in order, adds bookmarks for charts and
sections, and prints a visible result for each chart. An unresolved title prints
`not found "<title>"`, skips that chart, and continues rendering.

`lookup` prints the discovered source and extracts the matched page range. An
unresolved title prints `No matches found for "<title>"` and creates no file.

### Messages

- Default: found / not found per chart; per title once, competing candidates
  at the chosen rank, files taken from more than one folder (paths in the
  file-system view), and the substitutions taken (`Heartaches: Piano,
  Guitar, Bass, Drums → Concert`); missing parts.
- `-v`: also the configuration file used and the file chosen for each part.
- `-vv`: also every candidate (library, file, part, variations, free tags)
  and the naming rules that fired.

### Page numbering

When `page` is set, first interpret it as a decimal PDF page label. If the
label is unavailable, treat it as a 1-based physical page number. If `length`
is omitted while `page` is set, include one page. If both are omitted, include
the whole PDF. A part-index list of ranges is included in list order under
one bookmark.

### Output

- Produce one combined PDF per `--part` option. A part other than Concert
  writes `<stem>.<Part>.pdf`; a combined value writes `<stem>.<A>-<B>.pdf`;
  the rhythm section writes `<stem>.RS.pdf`. When `-o` already ends in the
  part or an alias of it (`.rs` for Rhythm), it is used as given.
- Add top-level bookmarks for flat chart entries.
- Add nested bookmarks for sections and their bodies.
- A title with several files gets a title bookmark with one child per file,
  named by the file's part (`Bass`, `Piano/Guitar`); a title with one file
  stays flat.
- Point an empty section at page 0.

### Errors

- Raise visible errors for missing library paths, malformed configuration,
  malformed indexed PDF JSON, malformed YAML collections, circular collection
  includes, missing source PDFs, invalid page ranges, and library records that
  omit required indexed-PDF fields or attach indexed-only fields to a directory
  library or YAML collection.
- Raise visible errors for both `tunery.yaml` and `.tunery.yaml` in one
  directory, unknown part names, and a `--test` title not in the layout.
- Include the layout path and available line/column information in YAML parsing
  errors.
- Include the layout path in schema validation errors.

## Backlog

- Multi-file page splits concatenated as one part (`P1`/`P2`, `_pg 2`).
- Group folders (`Bb Parts`) add their part to everything below.
- Sibling-prefix grouping for name texts.
- Rest of the big-band naming profile: trailing key, big-band instruments
  and codes (`tbn1`, `Saxes`), arranger, arrangement version; two-letter
  initials and camelCase titles (`FlyingHomeRHYTHM`).
- Durable named parts in the project configuration.
- Forced per-item metadata in library records; later, groups selected by
  pattern.
- Library exclusion list (`.gitignore` analog) beyond `_archive`.
- Inspection command listing discovered titles, parts and free tags.
- `lookup --part`.
- Per-patch interaction tests (new path vs. keys, collections, `file:`,
  mappings, links).
- Full score: every part of every title in one file.
