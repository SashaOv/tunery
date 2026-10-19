# Tunery

Tunery builds printable PDF setbooks from existing sheet-music PDFs. A layout
lists the tunes; libraries supply the charts, from folders of PDFs or from
JSON-indexed songbooks. Tunery picks the part each player needs.

## Configure libraries

Add an ordered `tunery.yaml` (or `.tunery.yaml`) file to the project:

```yaml
- library: indexes/real-book.json
  source: books/Real Book.pdf
  shift: 0
  match: fuzzy
- library: SheetMusic
  match: fuzzy
```

- Later libraries have higher priority, so list real books first and your
  own charts last.
- `source` is required for indexed PDF libraries; `shift` is optional and
  defaults to `0`. `match` is `exact` (default) or `fuzzy`.
- Paths are relative to the configuration file.

Tunery finds the configuration by walking up from the layout's folder (from
the working directory for `lookup`). A folder that is a symlink, such as a
link into Google Drive, is treated as part of the tree it is linked from:
walking up from `GDrive/Shows/gig` reaches the folder holding `GDrive`.
Without a configuration, the layout's folder is the library.

A `library` may also point to a YAML collection file holding more library
records, with paths relative to that file. A `match` set on the reference is
a default for inner records that don't set their own:

```yaml
- library: /path/to/rb-index/tunery.yaml
```

An indexed PDF library names one source PDF and its chart ranges:

```json
[
  {"title": "Autumn Leaves", "page": 39},
  {"title": "Song B", "page": 20, "pages": 2}
]
```

### Folder libraries

A folder library is searched recursively, following symlinks:

- `_archive` folders are skipped;
- folders starting with `_` (such as `_Petes`) are searched, but their names
  are ignored;
- a title matches the start of a file name or of its folder name
  (`Autumn Leaves/Autumn Leaves - Bass.pdf`, `Morning.Bb.pdf`).

What follows the title names the part (`Bass`, `Tenor Sax`, `Bb`, `Rhythm`,
`Lead Sheet`, `Piano:Guitar`) and its variations: score key (`Concert F`,
`(in F)`), version (`v2`), and anything else.

## Render a setbook

Create a layout such as `gig.yaml`:

```yaml
- config:
    - library: extra-charts
    - part:
        full: Concert, Trumpet, Tenor Sax, Rhythm
        norm: Tenor Sax
- section: Set 1
  body:
    - title: Giant Steps
    - title: Norwegian Wood
      key: F
      part: {Tenor Sax: Horn 1, Trumpet: Horn 2}
    - title: I Love Lucy
      tag: v2
    - title: Si Tu Vois Ma Mere
      file: ../Vault/Instrumentals/Si Tu Voir Ma Mere
    - file: Lennie's Pennies.pdf
      notes: Watch the ending
```

- Layout libraries are resolved relative to the layout and take priority
  over project libraries. A `part:` item names part selections.
- On a chart, `part:` (or `parts:`) says which part this tune uses for a
  requested one, `key:` picks a score key, and `tag:` (or `tags:`) prefers
  charts carrying a tag. `file:` names a PDF, or a folder whose PDFs are
  this tune's charts.

```console
tunery render gig.yaml                       # concert book: gig.pdf
tunery render gig.yaml --part "Tenor Sax"    # gig.Tenor Sax.pdf
tunery render gig.yaml --part norm --part rs # one book per --part
tunery render gig.yaml --part "Bass,Piano"   # both parts in one book
tunery lookup "Giant Steps"
```

- **Parts:** when a tune has no chart for the requested part, Tunery
  substitutes: an instrument falls back to its key group (Tenor Sax → Bb),
  then to the concert chart or lead sheet; Piano, Bass and Drums fall back
  to a Rhythm chart first, and Guitar to the Piano chart.
- **Rhythm section:** `Rhythm`, `RS` or `Rhythm Section` means
  `Piano,Guitar,Bass,Drums`.
- **Several charts for one tune** are nested under the tune's bookmark,
  named by part. `--explicit-bookmark-hierarchy` prefixes them with `_ ` for
  readers that show one bookmark level.
- **Warnings** report missing parts, competing charts and charts taken from
  more than one folder. `-v` adds the configuration used and the chart and
  substitution for each part; `-vv` adds every candidate.
- `--base DIR` resolves relative command-line paths against `DIR` instead of
  the working directory.

A plain list of titles works as a layout too: a `.txt` file with one title
per line, or a `.csv` file with a `Title` column.

```console
tunery render setlist.txt --part Bass
tunery render setlist.csv
```

Run `tunery --help` for all options.

## Developing

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Install dependencies with `uv sync`. `viewpath` comes from
  [kitbag](https://github.com/SashaOv/kitbag); to hack on it locally, point
  its entry in `[tool.uv.sources]` at `{ path = "../kitbag/viewpath", editable = true }`.
- Run `uv run build.py gate` before committing: it lints with ruff and runs
  the tests.
- Behavior is specified in `docs/SPEC.md`; changes in progress are planned
  in `docs/todo/`.

## License

Licensed under the Apache License, Version 2.0.

Copyright (C) 2024-2025 Sasha Ovsankin
