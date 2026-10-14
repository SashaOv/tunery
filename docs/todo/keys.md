---
refines: ../SPEC.md
---

# Key signatures parameter
This is a change spec to add key signature option to `render` function.

The command line parameter is '--keys' or '-k'. The definition and implementation of parser
should be the same as in pdfscore script, see /Users/sasha/personal/kitbag/pdfscore/pdfscore.py:614.

The logic of per-key signature edition is the same as in /Users/sasha/personal/niva/notation/cli/src/main.rs:166:
- If none specified, use "Concert" (C)
- Change the lookup of the files: add the name of the key signature as file suffix before the '.pdf' extension
- Concert editions can have "Concert" suffix or no suffix at all.

# Implement (built 2026-09-25; all items below are as-built)

## Scope
- `keys` added to `render` only (Python API + `render` CLI command). The
  `lookup` command is out of scope.

## CLI / API shape
- `tunery/main.py` `render_command` mirrors `pdfscore.py:614`:
  `keys: list[str]` via `Parameter(["--keys", "-k"])`, default `[]`.
  Cyclopts collects repeats; comma-splitting happens in expansion, same as
  pdfscore `expand_keys`.
- `tunery/render.py` `render(layout_path, output, keys: list[str] | None = None)`.
  `None`/`[]` means concert-only (previous behavior, fully backward compatible).

## Key expansion (`tunery/keys.py`, copied from pdfscore, not imported)
- `_canonicalize_key_token` / `normalize_key_name` / `KEY_ALIASES` /
  `KEY_NUMBERS` semantics copied from `pdfscore.py:88-282` (no cross-repo
  dependency): strip, collapse whitespace, `♯/♭` + `sharp/flat` handling,
  case-insensitive alias match, visible `ValueError` on invalid key.
- Each raw value is split on `,`, stripped, empties dropped, canonicalized
  (`bb` -> `Bb`), deduped preserving order. Empty result -> `["C"]` (concert,
  per niva `DEFAULT_EDITION = "C"`).
- `-k Concert` (any case) is accepted as an alias for `-k C`; the canonical
  key stays `"C"`, while `"Concert"` remains the concert filename suffix.

## Lookup semantics (niva `main.rs:166` + `edition_output_path` adapted)
- `key: str` is threaded through `process_file_entry` ->
  `LibraryCollection.lookup_title/lookup_file` -> `DirectoryLibrary`.
- `DirectoryLibrary` is the only place the suffix applies:
  - Non-concert key `K`: try `<stem>.<K>.pdf` only (e.g. `Song.Bb.pdf`).
    No fallback to the concert file; a missing suffixed PDF prints
    `not found "<title>"` for that key and rendering continues.
  - Concert key (`C`): try `<stem>.Concert.pdf` first, then `<stem>.pdf`
    (spec: "Concert editions can have Concert suffix or no suffix at all").
    Exact-then-fuzzy matching applies to the suffixed stem the same as today.
  - `lookup_file("Song.pdf")` with key `K`: try `Song.<K>.pdf` first, then
    (concert only) the plain name; the relative-to-layout fallback in
    `resolve_path` inserts `.<K>` before `.pdf` (`Song.Bb.pdf`), concert
    tries the `.Concert` variant (when that file exists) then the plain path.
- `file:` entries specifying `page`/`length` address the named PDF directly
  and skip key edition handling entirely.
- `IndexedPdfLibrary.lookup_*` ignores the key (same source/page range for
  all keys): indexed books are single concert-pitch sources; per-key
  transposed book PDFs are out of scope.
- Display titles/bookmarks keep the un-suffixed stem across keys; only the
  source file changes.

## Output fan-out (one setbook per key, like niva editions / pdfscore jobs)
- Every key derives its output from the resolved output path stem:
  concert -> `<stem>.pdf` (output as given), others -> `<stem>.<Key>.pdf`
  (e.g. `-o book.pdf -k C,Bb` -> `book.pdf` + `book.Bb.pdf`;
  `-o Combo3.pdf -k Bb` -> `Combo3.Bb.pdf`).
  Mirrors `edition_output_path` (`main.rs:65`, `render.rs:233`) and pdfscore
  `build_export_jobs` (`pdfscore.py:408`).
- Per-chart results are prefixed with `[Key]` when rendering multiple keys
  (niva `show_edition` equivalent), so `not found` lines are attributable.
  A key whose charts all miss produces no output file (empty `Composer`
  saves nothing).

## Tests (`tests/test_keys.py`, plus `tests/test_main.py` updates)
- Unit: key expansion/canonicalization (`bb, Eb, F-sharp` -> `Bb, Eb, F#`;
  `♭` handling; `Concert` alias; comma-split; dedupe; empty -> `["C"]`;
  invalid key errors); `keyed_filename`; `output_path_for_key`.
- Render: default render uses concert file and prefers `Song.Concert.pdf`
  when both exist; `-k Bb` uses `Song.Bb.pdf` with unchanged bookmark;
  `-k C,Bb` produces two outputs with expected page counts and `[Key]`
  prefixes; missing `Song.Eb.pdf` prints `[Eb] not found "Song"` and writes
  no `gig.Eb.pdf`; `file:` without pages uses the key edition, with
  `page`/`length` uses the plain file.
- CLI: `-k`/`--keys` repeat + comma forms forwarded to `render`.
