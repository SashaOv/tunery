---
refines: ../SPEC.md
---

# Library-Based Workflow

Layouts name titles; libraries supply the sheet music. A library discovers
its items' metadata (title, part, variation) from file and folder names.
Rendering looks titles up, picks the requested part, and builds the setbook
structure on top. `file:` layouts keep working unchanged.

## Concepts

- **Title:** the tune. A title appears at most once per layout.
- **Part:** what one player reads, in order of specificity:
  - instrument + role (`Alto Sax 2`, `Horn 1`; a role such as `Horn 1` may be
    played by different instruments);
  - instrument (`Piano`, `Tenor Sax`, `Vocal`);
  - group (`Concert`, `Bb`, `Eb`, `Bass Clef`, `Rhythm`);
  - kind (`Lead Sheet`, `Score`, `Lyrics`, `Solo`, `Changes`).
- **Variation:** what distinguishes two items of the same title and part.
  This covers score key (`Concert F`, `G Major`), arranger, and free tags
  (`JBM Arrangement`, `v2`). Variations are ignored unless selected;
  ambiguity is reported.
- **Library type:** `combo` (default) or `big band`. Sets rule priorities
  (below) and the default part of untagged items: Concert lead sheet for
  combo, full score for big band.

## Configuration

- **Discovery:** start at the base directory (`--base`, default: the logical
  working directory, as `file:` resolution already uses) and walk up at most
  eight parents. In each directory look for `tunery.yaml` and `.tunery.yaml`.
  The first directory holding one wins; holding both is an error. Configs are
  not merged. Without one, use the base directory as a directory library and
  print the existing notice.
- Library records keep their fields (`library`, `source`, `shift`, `match`)
  and gain `type`. Forced per-item metadata in the record overrides
  everything derived from names (backlog).
- Libraries are searched last to first (later records win), unchanged.

## Discovery

- **Items:** every PDF below a directory library, recursively, following
  symlinks without revisiting a real directory.
- **Folder roles:**
  - *excluded* — `_archive`: never scanned (treated as deleted);
  - *transparent* — name matches `_.*` (`_Petes`): contents scanned, name
    ignored;
  - *group* — whole name is a part (`Bb Parts`): name adds that part to
    everything below (backlog);
  - *title* — everything else.
- **Name texts** of an item: its stem and its innermost title folder. Lookup
  matches a layout title against either one.

## Interpretation

- **Normalization:** lowercase, `’`→`'`, apostrophes dropped, punctuation and
  `_` → space, compared *squashed* (spaces removed): `SoWhat` = `So What`,
  `Mo’ Better Blues` = `Mo Better Blues`. Stem cleanup: unwrap
  `Finale 2004a - [<name>]`, drop a leading `Copy of `/`Copy (2) of `, drop a
  trailing `(1)`.
- **Boundaries:** hard are ` - `, `-`, `.`, `,`, `( )`, `[ ]`, `:`; soft is a
  space. A title may end at a soft boundary only when everything after it is
  tags (`Morning Piano` is a Morning part, `Morning Dance` is not).
- **Title match happens at lookup.** Layout title `T` matches a name text
  that starts with `T` at a boundary: exact squashed first, then fuzzy
  (ratio ≥ 0.9) for `match: fuzzy` libraries. What follows `T` in the stem is
  interpreted as tags. That is why `Partido Alto` keeps "Alto" and
  `Fascinating Rhythm` keeps "Rhythm". When only the folder matches, the
  stem's tags are read right to left up to the first word that isn't a tag.
- **Rules:** the tag text is read right to left. At each position, rules are
  tried in priority order and the longest span wins (first-match; no
  whole-name scoring). Library type sets each rule's priority; 0 disables it.
  Equal priority → rule order.

  | Rule | Matches | Combo | Big band |
  |---|---|---|---|
  | Explicit score key | `Concert <key>`, `in <key>`, `<key> major/minor`, `<key>m` | on | on |
  | Instrument | vocabulary below, incl. roles `Horn N`, `Alto Sax 2` | on | on |
  | Group | `Bb`, `Eb`, `Concert`, `C Instruments`, `… Instr`, `… part`, `Melody` | high | low: after `.` or next to an instrument/kind |
  | Kind | `Lead Sheet`, `Score`, `Lyrics`, `Solo`, `Changes`, `Structure`, `Harmony` | on | on |
  | Page split | `Pg N`, `P1`, `_pg 2` | on | on |
  | Noise | `original`, `rescan`, `v2`, `0`, `org`, `iRealPro`, track numbers, fillers (`part`, `in`, `with`, `the`, `n`, `and`, `&`) | on | on |
  | Moved article | trailing `, A` / `, The` | on | on |
  | Trailing key | lone `<key>` last after ` - `, or in `( )`/`[ ]` | 0 | on |
  | Big-band instruments | sections (`Saxes`, `Trumpets 1 n 2`), doubles (`F Horn`, `Tuba`, `Baritone TC`, `Violins`), codes (`tbn1`, `bsx`, `as1`) | 0 | on |
  | Arranger | trailing `-Name` / `(Name)` no other rule takes | 0 | on |
  | Version | trailing number after title, arranger or key | 0 | on |

- **Vocabulary** (instrument → group): Soprano Sax (soprano) → Bb; Alto Sax
  (alto, alt, as) → Eb; Tenor Sax (tenor, ten, ts, tenorsax) → Bb; Bari Sax
  (bari, baritone, bsx) → Eb; Clarinet (clarinet in bb) → Bb; Bass Clarinet
  (bass clr) → Bb; Trumpet (tpt, tp, trumpet in bb) → Bb; Flugelhorn (flugel)
  → Bb; Trombone (bone, tbn) → Bass Clef; Flute → Concert; Vocal (voice, vx)
  → Concert; Piano (pno, keys, electric piano), Guitar (gtr, electric
  guitar), Piano+Guitar (piano-guitar, Piano:Guitar), Bass (bss, electric
  bass), Drums (drm), Percussion → Rhythm. `_` joins words
  (`Bass_Clarinet`, `Alto_Saxophone`).
- **Part of an item:** the rightmost instrument (+ role). If there is none,
  the group, then the kind, then the library's default part. An earlier,
  unrelated instrument is a variation (`All of Me (Vocal)-Piano` → Piano,
  variation "vocal"). A score key is a variation.
- **Can't interpret:** words no rule takes are free tags (`lead-sheet-d`,
  `skin p`). The item stays reachable by its name texts; nothing is guessed.

## Selection

- **Part selection:** `render --part <Part>`; default `Concert`.
- **Substitution chains:** Instrument N → Instrument → section → group →
  Concert. Piano/Guitar → Piano+Guitar → Rhythm → Concert. Bass → Rhythm →
  Concert. The Concert step takes Concert or untagged, then Lead Sheet, then
  Vocal/Melody, then other concert-group instruments.
- **Order:** libraries in priority order; the first library with an item on
  any step of the chain wins, at its earliest step. So a Vault Concert chart
  beats a real-book page for a Piano request.
- **Ties** within one library and step: fewest variation/free tags, then path
  order. Print a visible warning naming the candidates.
- **Indexed PDF libraries** are concert-only sources: they answer any part
  request with the indexed pages, as today.
- **Keys:** `-k K` selects part group `K` with no substitution (the keys
  feature's semantics). Output naming is unchanged: `<stem>.<K>.pdf`.
- **Output:** `--part P` other than Concert writes `<stem>.<P>.pdf`.
- **Trace:** `-v` prints per title the candidates (library, file, part,
  variation and free tags), the chain step taken, and the rules that fired.

## Open questions

- Q16 Groups: confirm Trombone → Bass Clef, and Bass in Rhythm only.
- Layout-level part selection and per-title overrides (`part: Horn 1`): exact
  syntax.
- Default for `--base` when the layout lives in another tree: is the working
  directory enough, or should the layout directory be searched after it?

## Facts

Surveyed 2026-10-03 and 2026-10-14 by session-scratchpad prototypes (not in
the repo).

| Library | Type | PDFs | Naming |
|---|---|---|---|
| JBM `Vault` | combo | 202 | `Category/Title[/Title - Concert <Key>]/<slug>-<Instrument>.pdf` |
| Crosswinds `SheetMusic` (+ `_Petes`, `Bb Parts`) | combo | 212 | `Title/Title.<part>.pdf`; `_Petes/…/Title - Part.pdf` |
| SOJ `MasterList` | big band | 398 | `Title-Arranger.pdf` or `Title-Arranger/<Part>.pdf` |
| SOJ `cache/repo` | big band | 2228 | `Title[ - Key].zip/<Part> Pg N.pdf` |

- **Same text, different meaning by type:** trailing ` - Bb` is a group in
  combo (`Soft Impressions - Bb`) but a score key in big band
  (`Kansas City - Bb`; 180 repo names end in ` - <key>`). `(Word)` is a
  part/variation in combo (`(Vocal)`, `(Bb Part)`) but an arranger in big
  band (`(Mason)`, `(Wolpe) 2`).
- **First-match suffices:** of 435 key-like tokens in all five libraries, 333
  are settled by position, 61 are title words, 34 are one multi-word span
  (`Ceora Bb Tenor`, `Mack Bb Solo`), 7 need a rule variant or stay free
  tags. None needs whole-name scoring.
- **Sibling prefixes** (siblings sharing a title prefix, suffixes all tags)
  give 0 false groups over 812 combo and big-band PDFs. They need: pure-tag
  stems excluded, largest group then longest prefix, no free words in
  suffixes.
- **Numbered roles exist in combo:** `Norwegian Wood - Horn 1/2`
  (`26-09-15.norm.yaml` selects `Horn 1`), so roles are not big-band only.
- **Folder-only names:** `_.*` matches only `_Petes` and `_archive` (both
  `SheetMusic`). Acronym titles (`fmttm`, `gd`, `cp`, `lihts`) occur only in
  big-band libraries.
- **JBM 26-10-21** (2026-10-14), simulated with the rules above, `--part
  Piano`, Vault fuzzy:
  - 15 of 22 titles resolve from Vault.
  - Of the 13 titles also hand-picked in `26-10-02.yaml`, 12 give the same
    file. `Ain't Misbehavin` is a 3-way tie (`Concert F`, `Concert Bb`,
    `Tunes for Michelle`); 26-10-02 used Concert F.
  - Ties: `Mo Better Blues` (`Piano` / `Piano_(1)`), `All of Me` and
    `Girl From Ipanema` (copies in `Tunes for Michelle`).
  - The other 7 aren't in Vault under any name. rb-index (last resort) has 6;
    `It Don't Mean Thing` is a layout typo.
  - `Si Tu Vois Ma Mare` needs `match: fuzzy` (folder `Si Tu Voir Ma Mere`,
    stem `si-tu-vois-ma-mere-…`).
- **Config discovery today** resolves the layout directory physically. From
  `JBM/GDrive/Shows/26-10-21` (`GDrive` → Google Drive), after the author
  removed the Drive-side `tunery.yaml`, it finds no config and renders
  nothing from Vault. `apps/JBM/tunery.yaml` (rb-index, then `./Vault`) is
  reachable only through the logical tree.

## Builds

### LIB4: Title-only combo layouts render the requested part from a nested folder library

Status: planned, awaiting author checkpoint

**Improvement:** run from `apps/JBM`,
`tunery render GDrive/Shows/26-10-21/26-10-21.yaml --part Piano` renders 15
titles from the JBM Vault, each with the part chosen by the chains above.
Six come from rb-index, and the run warns about ties. Today the same layout
finds no config, or, with the old Drive config, takes everything from real
books.

**Prerequisite (not a LIB4 patch):** commit the existing uncommitted keys
feature (`tunery/keys.py`, `tests/test_keys.py`, `library.py`, `render.py`,
`main.py`, `tests/test_main.py`; suite: 76 passed on 2026-10-14) as its own
commit, per Q1. The deletions of `AGENTS.md`, `CLAUDE.md` and
`docs/GUIDELINES.md` are the author's and stay out of it.

**Patches:**

- **LIB4.1** Discover configuration from `--base`. Add `--base` (default:
  logical working directory) to `render` and `lookup`. Search `tunery.yaml`
  and `.tunery.yaml` up from it; both in one directory is an error. Tests:
  each file name, both present, walk through a symlinked directory, notice
  when none.
- **LIB4.2** Interpret item names (`tunery/parts.py`): normalization,
  boundaries, title match with remainder, combo-profile rules right to left,
  vocabulary, part/variation/free tags. Unit tests on the combo names in
  Facts and every 26-10-02 Vault file name.
- **LIB4.3** Discover directory libraries recursively: symlinks without
  cycles, `_archive` excluded, `_.*` transparent, name texts = stem +
  innermost title folder. Existing flat-library tests pass unchanged.
- **LIB4.4** Select parts: `render --part`, substitution chains, library
  priority, tie warnings, `-k` as group without substitution,
  `<stem>.<Part>.pdf` output, indexed libraries as concert sources. Fixture:
  a temp tree with the Vault file names of the 15 titles. It asserts each
  pick from Facts and the tie warnings.
- **LIB4.5** Trace with `-v`: per title, candidates, chain step and fired
  rules.

**Acceptance:**

- *Scripted (pytest, the existing gate):* all existing tests, including the
  keys tests, pass unchanged; the LIB4.1–LIB4.5 tests pass.
- *Live (executor):* from `/Users/sasha/personal/wip/apps/JBM`, run
  `tunery render GDrive/Shows/26-10-21/26-10-21.yaml --part Piano -o
  <scratchpad>/26-10-21.Piano.pdf`. Observe:
  - 15 titles found under `Vault/` with the files listed in Facts;
  - 6 found in rb-index books;
  - `not found "It Don't Mean Thing"`;
  - tie warnings for `Ain't MisBehavin'`, `Mo Better Blues`, `All of Me`
    and `Girl From Ipanema`;
  - a PDF with 21 charts.
  With `-v`, each title lists its candidates.

**Author action required at completion:**

- Add `match: fuzzy` to the `./Vault` record in
  `/Users/sasha/personal/wip/apps/JBM/tunery.yaml`; without it, `Si Tu Vois
  Ma Mare` is not found (live clause: 15 Vault titles).
- Review the rendered PDF musically, and choose the `Ain't Misbehavin` key
  (a `file:` override in the layout until variation selection exists).

## Backlog

- Library `type` field with the big-band profile (rules marked big band
  above).
- Group folders (`Bb Parts`) add their part to everything below.
- Sibling-prefix grouping for name texts (rules in Facts).
- Layout-level part selection, named editions (`norm`, `rs`) and per-title
  part overrides; durable selections in config.
- Several parts per run, one output per part; full score with all parts.
- `--explicit-bookmark-hierarchy` for the `_ ` bookmark prefix.
- Variation selection (score key, arranger, tags).
- Forced per-item metadata in library records; later, groups selected by
  pattern.
- Multi-file page splits concatenated as one part (`P1`/`P2`, `_pg 2`).
- Parts inside multi-part score PDFs (`scores.json`, schema may change).
- Library exclusion list (`.gitignore` analog) beyond `_archive`.
- Inspection command listing discovered titles, parts and free tags.
- `lookup --part`.
- Big-band best-effort: acronym and camelCase titles (`fmttm…`, `gd…`,
  `FlyingHomeRHYTHM`).
