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
  (`JBM Arrangement`), and versions (`v2`). Variations are ignored unless
  selected (`key:`, `tag:`); ambiguity is reported.
- **Library type:** `combo` (default) or `big band`. Sets rule priorities
  (below) and the default part of untagged items: Concert lead sheet for
  combo, a jumbo file (every part in sequence; LIB7) for big band. A conductor
  score (all instruments on one page) is a part of its own in both.
- **File-system view** (author, approved 2026-10-15):
  - The **base** directory is the root of tunery's view of the file system.
    It is `--base` if given, else the logical working directory (`$PWD`).
    Relative paths on the command line are joined to it.
  - Paths are navigated lexically: `..` is collapsed and directory symlinks
    are never resolved.
  - A directory symlink (`GDrive`, `Vault`) is a **link**: a window into
    another tree, usually cloud storage. Going down through a link enters
    that tree. Going up from inside it returns through the link to the base
    hierarchy, never to the linked tree's physical parents.
  - Example: from `crosswinds/GDrive/Shows/26-09-15`, up is
    `GDrive/Shows`, `GDrive`, `crosswinds`, `personal`, …; never
    `My Drive/Ongoing`.
  - Messages show paths in this view (`GDrive/SheetMusic/…`), not resolved
    ones (`.shortcut-targets-by-id/…`).

## Configuration

- **Discovery:** start at the layout's directory, in the file-system view,
  and walk up to the filesystem root. Commands without a layout (`lookup`)
  start at the base. In each directory look for `tunery.yaml` and
  `.tunery.yaml`. The first directory holding one wins; holding both is an
  error. Configs are not merged. Without any config, the library is the
  layout's directory (the base for `lookup`), and the existing notice is
  printed.
- A config that is itself a symlinked *file* resolves its relative paths
  against its real location (unchanged). A collection file reached through a
  linked folder stays in the file-system view, like a project config.
- Library records keep their fields (`library`, `source`, `shift`, `match`)
  and gain `type` (`combo`, the default, or `big band`). A collection
  reference passes `type` on like `match`. Forced per-item metadata in the
  record overrides everything derived from names (backlog).
- **Part index** (big band): a record's `index:` lists `{title, file?,
  parts}` entries, matched by normalized title (fuzzily in fuzzy
  libraries); part values are `page`, `[page, length]` or a list of
  ranges. Each entry resolves its file (explicit `file:`, else the title's
  jumbo file) and contributes one candidate per part at that part's own
  step.
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
  `Finale 2004a - [<name>]`, drop a leading `Copy of `/`Copy (2) of ` (kept
  as a noise word), drop a trailing `(1)`.
- **Boundaries:** hard are ` - `, `-`, `.`, `,`, `( )`, `[ ]`, `:`; soft is a
  space. A title may end at a soft boundary only when everything after it is
  tags (`Morning Piano` is a Morning part, `Morning Dance` is not). After a
  hard boundary, words no rule takes are free tags (`mo-better-blues--JBM
  Arrangement-Piano` → Piano, free tags "JBM Arrangement").
- **Title match happens at lookup.** Layout title `T` matches a name text
  that starts with `T` at a boundary: exact squashed first, then fuzzy for
  `match: fuzzy` libraries. Fuzzy compares `T` with each word prefix of the
  name text (ratio ≥ 0.9, never a substring further in), so the words after
  the matched prefix still give the part (`Si Tu Vois Ma Mare` →
  `si-tu-vois-ma-mere-…-lead-sheet-C_Major`, Lead Sheet). What follows `T` in the stem is
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
  | Noise | `original`, `rescan`, `0`, `org`, `iRealPro`, a leading `Copy of`, track numbers, fillers (`part`, `in`, `with`, `the`, `n`, `and`, `&`) | on | on |
  | Version | `v2`, `v3`, …: a variation tag | on | on |
  | Moved article | trailing `, A` / `, The` | on | on |
  | Trailing key | lone `<key>` last after ` - `, or in `( )`/`[ ]` | 0 | on |
  | Big-band instruments | sections (`Saxes`, `Trumpets 1 n 2`), doubles (`F Horn`, `Tuba`, `Baritone TC`, `Violins`), codes (`tbn1`, `bsx`, `as1`) | 0 | on |
  | Arranger | trailing `-Name` / `(Name)` no other rule takes | 0 | on |
  | Arrangement version | trailing number after title, arranger or key | 0 | on |

- **Vocabulary** (instrument → group): Soprano Sax (soprano) → Bb; Alto Sax
  (alto, alt, as) → Eb; Tenor Sax (tenor, ten, ts, tenorsax) → Bb; Bari Sax
  (bari, baritone, bsx) → Eb; Clarinet (clarinet in bb) → Bb; Bass Clarinet
  (bass clr) → Bb; Trumpet (tpt, tp, trumpet in bb) → Bb; Flugelhorn (flugel)
  → Bb; Trombone (bone, tbn) → Bass Clef; Flute → Concert; Vocal (voice, vx)
  → Concert; Piano (pno, keys, electric piano), Guitar (gtr, electric
  guitar), Bass (bss, electric bass), Drums (drm), Percussion → Rhythm; Bass
  Trombone → Bass Clef; Saxophone (sax, saxophone: a sax of unknown key, so
  `Tenro Sax part` is not taken for the concert chart) → no group. `_` joins
  words (`Bass_Clarinet`, `Alto_Saxophone`).
- **Dual-instrument scores:** a name that joins two instruments in one
  segment (`Piano:Guitar`, `piano-guitar`, `PianoGuitar`, `pnoguit`) is one
  score both instruments read. The item has both parts, Piano and Guitar. A
  multi-word instrument (`Bass Clarinet`) is not a dual, and an instrument
  in parentheses is a variation, not half of a dual.
- **Score formats** (big band): a file can hold one part (`Hay
  Burner-Nestico/Bass.pdf`, found by name as today), every part in sequence
  (**jumbo**, named for no part: `Bandstand Boogie-Lowden.pdf`), or a few
  related parts in sequence (**combined set**, named for their group:
  `Rhythm Parts.pdf`). A jumbo or combined-set file is never reached by a
  substitution chain, so it never stands in for a part. A conductor score
  (`conductor`, `score`, `condensed score`) is a part of its own
  (`Conductor`): recognized and listed, never substituted for another part.
  An arranger needs no rule: `-Lowden` is a free tag.
- **Part of an item:** the rightmost instrument (+ role), or both instruments
  of a dual-instrument score. If there is none, the group, then the kind,
  then the library's default part (Concert lead sheet for combo, full score
  for big band). An earlier, unrelated instrument is a
  variation (`All of Me (Vocal)-Piano` → Piano, variation "vocal"). A role
  in parentheses right after an instrument is the role that instrument
  plays: `tenor (horn 2)` is Tenor Sax as Horn 2, matching both. A score
  key is a variation.
- **Can't interpret:** words no rule takes are free tags (`lead-sheet-d`,
  `skin p`). The item stays reachable by its name texts; nothing is guessed.

## Selection

- **Part selection:** `render --part <Part>`; default `Concert`.
  - Each `--part` option produces its own output (`--part Bass --part Piano`
    → two files).
  - A comma-separated value combines parts in one output
    (`--part Bass,Piano` → one file).
- **Rhythm section:** `Rhythm`, `RS` and `Rhythm Section` select the
  combination `Piano,Guitar,Bass,Drums`, also inside a comma list
  (`Concert,Trumpet,Tenor Sax,Rhythm`). Each member follows its own chain,
  so a title without rhythm parts falls back to concert-key instruments and
  the lead sheet.
- **Named parts** (author, 2026-10-15): a `part:` item in the layout's
  `config:` list defines names for part selections, used like built-in
  parts:
  ```yaml
  - config:
      - part:
          full: Concert, Trumpet, Tenor Sax, Rhythm
          norm: Tenor Sax
  ```
  `--part norm` writes `<stem>.norm.pdf`. An unknown `--part` name is a
  visible error.
- **Per-title part mapping** (author, 2026-10-15): a chart's `part:` maps
  requested parts to the parts this title uses for them. Keys are parts, so
  one mapping serves every named part that contains them:
  ```yaml
  - title: Norwegian Wood
    part: {Tenor Sax: Horn 1, Trumpet: Horn 2}
  - title: Funk in Deep Freeze
    part: {Tenor Sax: Flute}
  ```
  `part:`/`parts:` and `tag:`/`tags:` are both accepted, and may hold
  several values (author, 2026-10-15).
  Which instrument reads `Horn 1`/`Horn 2` is decided only here; no rule
  guesses it.
  A mapping key may also name a book from the layout's `part:` config
  (`parts: {norm: Flute}`): it replaces the title's parts when rendering
  that book and is ignored elsewhere. Part keys apply in every book.
- **Substitution chains:**
  - Instrument N → Instrument → section → group → Concert rank → Melody →
    Vocal.
  - Piano → Rhythm → Concert rank → Melody → Vocal.
  - Guitar → Piano → Rhythm → Concert rank → Melody → Vocal.
  - A Piano/Guitar score matches the Piano and the Guitar steps directly.
  - Bass → Rhythm → Concert rank → Melody → Vocal.
  - Drums → Rhythm → Concert rank → Melody → Vocal.
  - The Concert rank is Concert or untagged, then Lead Sheet, then a Rhythm
    chart, in that step order. Concert books then fall back to Melody, then
    Piano/Guitar (one rank), then Vocal. Flute no longer substitutes for
    Concert.
  - The "Rhythm" step matches items tagged `Rhythm` or `Rhythm Section`.
- **Combined parts:** each requested part resolves through its chain. If two
  parts resolve to the same file, it is included once. A part whose chain
  finds nothing for a found title is reported (`missing <part> for
  "<title>"`). A title with one resulting file keeps a flat bookmark. A title
  with several gets a title bookmark with one child per file, named by the
  file's part label (a dual score is `Piano/Guitar`):
  - All of Me
  - Autumn Leaves
    - Bass
    - Piano
- **Order:** libraries in priority order; the first library with an item on
  any step of any requested part's chain wins the title, at each part's
  earliest step. So a Vault Concert chart beats a real-book page for a Piano
  request. All parts of a title come from that library: a part it lacks is
  reported missing, never filled from a lower library, so one title's parts
  belong to one arrangement.
- **Ranks:** in a chain, an instrument and its key group are one rank: a
  Bb chart is as good as a Tenor Sax chart (likewise Alto Sax/Bari Sax and
  Eb, Trombone and Bass Clef). Concert, Lead Sheet and Rhythm charts are one
  rank, as are Piano and Guitar charts. Rhythm is not a key group, so Bass
  still ranks above a Rhythm chart.
- **Sources:** within the winning library, each folder holding candidates
  for a title is a source. The title's top source has the most requested
  parts found at their own step, then the fewest extra words, then comes
  first in path order.
- **Choice per part:** each part takes the best rank at which any source has
  a candidate. Among the candidates at that rank, the order is:
  1. those carrying the chart's `tag:`;
  2. the fewest substitution steps: a chart's position in the part's chain,
     0 for the part itself (`tenor (horn 2)` before `.Bb` for Tenor Sax;
     Alto Sax before Eb for Alto Sax 2). A chart's `part:` mapping is not a
     substitution: the mapped part is the one requested (author,
     2026-10-15);
  3. those in the top source;
  4. the fewest extra words (variations, free tags and noise words each add
     1: `Creepin.pdf`, 0, beats `Creepin.iRealPro.pdf`, 1);
  5. path order.

  Example: Poinciana's Bass is only in `_Petes/Poinciana`, which makes it
  the top source, so its concert chart wins over `SheetMusic/Poinciana`'s.
  Trumpet still takes `SheetMusic`'s `Poinciana.Bb.Emaj`, a better rank
  than any `_Petes` chart.
- **Warnings:** once per title, when a part had several candidates at its
  rank, or when the title's files come from more than one folder. The
  warning names the files used and the others (paths in the file-system
  view). This flags a stray source such as `SheetMusic/I Cant Help It`
  beside `_Petes`, and duplicates such as `I Love Lucy - Tenor Sax` vs
  `- Tenor Sax.v2`.
- **Indexed PDF libraries** are concert-only sources: they answer any part
  request with the indexed pages, as today.
- **Output:** `--part P` other than Concert writes `<stem>.<P>.pdf`; a
  combined value writes `<stem>.<A>+<B>.pdf` (`RS` stays `<stem>.RS.pdf`).
  When `-o` already ends in `.<P>` or an alias of `P` (`.rs` for Rhythm),
  it is used as given.
- **Paths in messages** follow the file-system view (Concepts).
- **Score folder per chart:** a chart's `file:` may name a folder (relative
  to the layout, like a PDF `file:`). The title's parts are then chosen
  among the PDFs in that folder only, without matching their names to the
  title, so an oddly named score is reachable without fuzzy matching:
  `- title: Si Tu Vois Ma Mare` / `file: ../../../Vault/Instrumentals/Si Tu
  Voir Ma Mere`.
- **Tag per chart:** a chart's `tag:` (`tag: v2`, or a list) prefers, part
  by part, candidates carrying all those variation or free tags (see Choice
  per part). If no candidate of the title carries them, warn `no <tag>
  version of "<title>"`.
- **Bookmark hierarchy:** `--explicit-bookmark-hierarchy` prefixes child
  (part) bookmarks with `_ ` (`_ Bass`), for readers that show only one
  bookmark level.
- **Score key per chart:** a chart's `key:` (`- title: Ain't MisBehavin'` /
  `key: F`) keeps only candidates whose score key matches. A key given
  without a mode matches any mode. If no candidate in a library matches, use
  all its candidates and warn `no <key> version of "<title>"`. The layout's
  `key` is the score key.
- **Verbosity:**
  - default: found / not found per chart, source and candidate warnings,
    missing parts, and one substitution line per title that substituted
    (`Heartaches: Piano, Guitar, Bass, Drums → Concert`);
  - `-v` / `--verbose 1`: additionally, the configuration file used, and per
    title the file used for each part;
  - `-vv` / `--verbose 2`: additionally, every candidate (library, file,
    part, variation, free tags) and the name-interpretation rules that
    fired.

## Open questions

- Q16 Groups: confirm Trombone → Bass Clef, and Bass in Rhythm only.
- Output name for combined values: `<stem>.<A>+<B>.pdf` is a proposal.

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
  - With `--part Rhythm` and the layout's `key: F` on `Ain't MisBehavin'`:
    - the same 15 Vault titles;
    - `Mack the Knife` → Piano, Guitar, Bass (`…Score-Piano`, `-Guitar`,
      `-Bass`) and `Straighten Up and Fly Right` → Piano, Guitar
      (`…-Piano`, `-Electric_Guitar`); every other title is one chart;
    - `Ain't MisBehavin'` → `Aint Misbehavin - Concert F/…-Piano`, no tie;
    - ties left: `Mo Better Blues`, `All of Me`, `Girl From Ipanema`;
    - Heartaches, Margaritaville, Si Tu Vois Ma Mare, Fascinating Rhythm,
      Steamroller Blues and Girl From Ipanema fall back to their concert
      chart; none of these titles has a Drums part in Vault, and only Mack
      the Knife has Bass.
- **Rhythm section vs `26-09-15.rs.yaml`** (2026-10-14). Applied to the parts
  available in `26-09-15.yaml`, the RS combination reproduces the hand-made
  rs layout for 22 of 23 titles. Examples: Poinciana → Concert + Bass,
  I Can't Help It → Piano + Bass + Drums, Mo Better Blues → Rhythm + Bass +
  Drums; titles with only Concert/Bb stay one flat chart. The exception,
  Norwegian Wood, adds its Piano:Guitar dual score, which both Piano and
  Guitar resolve to (included once). The Guitar → Piano step is needed for
  I Can't Help It, whose guitar player reads the piano part; otherwise its
  concert sheet would be added.
- **Crosswinds `26-09-15.1.rs.yaml`, `--part Rhythm`** (2026-10-15, after
  LIB4.9):
  - From the show folder, discovery stops at `Shows/26-09-15/tunery.yaml`, a
    symlink to rb-index, so only real books are used: 17 found, 6 not found.
    The author's `26-09-15.1.rs.pdf` (18 titles, no nested bookmarks) also
    comes from real books mostly.
  - `Shows/tunery.yaml` lists `../SheetMusic` before rb-index. Later records
    win, so real books shadow SheetMusic for 17 of 23 titles. The JBM
    config puts rb-index first.
  - With rb-index first (scratch config via `--base`), 21 of 23 titles match
    `26-09-15.rs.yaml`, and Norwegian Wood adds Piano/Guitar as designed.
    The misses: Creepin' takes `Creepin.iRealPro.pdf` (noise words don't
    count in the tie-break), and Poinciana takes `SheetMusic/Poinciana.pdf`
    with the `_Petes` bass part (mixed arrangements).
  - Child bookmarks are named by the requested part, not the file's. Mo'
    Better Blues shows "Piano" for its Rhythm chart, Norwegian Wood "Piano"
    for Piano:Guitar, and Poinciana "Piano" for the concert chart.
  - Tie warnings repeat once per rhythm member. They fire even when the
    tie-break decides (Red Clay `.C-sharp` vs `- concert`), and they list
    bare file names (`All of Me (Vocal)-Piano.pdf` twice).
  - Messages show resolved paths (`.shortcut-targets-by-id/1Aou…/Crosswinds/…`).
  - `-o …/x.rs.pdf --part Rhythm` writes `x.rs.Rhythm.pdf`.
- **Run from a parent folder** (2026-10-15): from `/Users/sasha/personal/wip`
  with `apps/JBM/GDrive/Shows/26-10-21/26-10-21.yaml`, discovery starts in
  `wip` and finds no config. The fallback library is then all of `wip`,
  searched recursively, so `All of Me` gets 12 candidates from the SOJ copies
  (`All of Me-Niehaus.pdf`, …). The only hint is the "Project configuration
  was not found" notice. The same happens from `/Users/sasha/personal/crosswinds`
  with `GDrive/Shows/26-09-15/26-09-15.1.rs.yaml`: no config is found above
  `crosswinds`, so all of it is scanned through the `GDrive` link. The
  layout's own walk would reach `GDrive/Shows/tunery.yaml`. Even so, 21 of 23
  titles match `26-09-15.rs.yaml`; Poinciana mixes arrangements and Norwegian
  Wood adds Piano:Guitar.
- **`SheetMusic/Bb Parts`** is a Google Drive shortcut into another shared
  folder (`Take Five Bb Tenor`, `Ceora Bb Tenor`, …). Discovery follows it
  like `_Petes/Crosswinds`. The author did not intend it to be in scope
  (2026-10-15); renaming or moving it is the author's call.
- **Norm and full books from titles** (2026-10-15, LIB4.9 code, Drive show
  folder, `Shows/tunery.yaml`), titles of `26-09-15.1.rs.yaml`:
  - `--part "Tenor Sax"` matches `26-09-15.norm.yaml` for 14 of 23 titles.
  - `--part "Concert,Trumpet,Tenor Sax,Rhythm"` matches `26-09-15.yaml` for
    11 of 23.
  - The differences:
    - `SheetMusic/Bb Parts` (`Take Five Bb Tenor`, `Ceora Bb Tenor`,
      `Mercy Mercy Mercy Bb Tenor`, `Red Clay Bb Tenor`) is a direct Tenor
      Sax hit, so it beats each title folder's `.Bb` chart, which both hand
      layouts use.
    - `The Cape Verdean Blues - tenor (horn 2)` reads as part Horn, so a
      Tenor Sax request falls back to SheetMusic's `.Bb`.
    - `I Love Lucy - Tenor Sax.v2` (hand layouts) loses to `_Petes`'
      unversioned `Tenor Sax`.
    - `Copy of Shaker Song - Tenor` ties with `Shaker Song - Tenor` and wins
      on path order.
    - In a combined value, `Rhythm` isn't expanded to the rhythm section,
      so Mo' Better Blues, Norwegian Wood and I Can't Help It miss their
      Bass/Drums/Piano parts.
    - Norwegian Wood's horns are `Horn 1`/`Horn 2`, which no Trumpet or
      Tenor Sax chain reaches.
    - Norm's two exceptions (Norwegian Wood → Horn 1, Funk in Deep Freeze →
      Flute) need per-title overrides.
    - The full book also takes `I Cant Help It.pdf` (concert, SheetMusic),
      which `26-09-15.yaml` lacks although its prompt asked for unsuffixed
      concert charts.
- **LIB5 review** (2026-10-15, `cache/lib5-review.md`). Three defects
  reproduced on tree `fd54ad6`:
  - `-k` bypasses part selection: score folders ignore it, and title
    entries still use the `<Title>.<K>.pdf` suffix lookup, so
    `Tune/Tune - Bb.pdf` isn't found while `Flat.Bb.pdf` is;
  - a collection file reached through a folder link resolves physically
    (`library.py`, collection load), unlike project configs;
  - chart `part:` mappings aren't validated (`{Tenor Sax: Fluet}` falls back
    to Concert silently).

  Two of the three are rules applied at one edge (part validation in
  `--part` expansion, the lexical-path rule in the project-config loader)
  that a new caller bypassed.
- **Concert book falls through to real books** (2026-10-15): without
  `--part`, Recado Bossa Nova comes from the Bob book, because SheetMusic has
  only `- Rhythm` and `-  Bb` charts (in `_Petes/Crosswinds`) and the Concert
  chain stops at Flute. A Rhythm chart is concert pitch.
- **Concert = Lead Sheet = untagged = Rhythm chart** (author, 2026-10-15),
  simulated by patching the chains on the live libraries (session
  scratchpad `sim_concert.py`):
  - Crosswinds concert book: Recado Bossa Nova, Mo' Better Blues, I Love
    Lucy, Jeannine, Jolene, Funk in Deep Freeze and Comin' Home Baby take
    their Rhythm chart (Funk no longer takes Flute; Recado and Comin' Home
    Baby no longer fall back to real books).
  - Concert then falls back to Melody, then Piano/Guitar, then Vocal
    (author). I Can't Help It's concert
    book takes `- Piano`, so no Crosswinds title comes from a real book.
  - For rhythm players the Rhythm chart stays ahead of Concert: with it
    after Concert, The Cape Verdean Blues' rs chart turned into the
    SheetMusic concert chart.
  - rs 22/23 and full 21/23 against the hand layouts. Full's Funk in Deep
    Freeze loses Flute (its Concert is now the Rhythm Section chart), and
    Cape Verdean adds SheetMusic's `.Bb` and `.pdf`.
  - JBM rhythm-section picks are unchanged (22/22), and its 17 `missing`
    lines go, since Bass and Drums now substitute to the Piano chart. JBM's
    concert book: All of Me and Quizas take `-Piano`; Girl From Ipanema and
    Steamroller Blues take `-Voice`.
- **Poinciana concert book** (2026-10-15) takes
  `SheetMusic/Poinciana/Poinciana.pdf`, which ties with
  `_Petes/Poinciana/Poinciana - concert.pdf` (both Concert, 0 steps; neither
  folder supplies another requested part) and wins on path order (warned).
  The author wanted the `_Petes` chart and moved `SheetMusic/Poinciana` to
  `_archive` (2026-10-15), so no rule change is needed. That also archives
  `Poinciana.Bb.Emaj.pdf`, the hand full book's Bb chart: full drops to
  21/23 with LIB5 code (20/23 simulated with LIB6.4).
- **cyclopts 4.20** (2026-10-15): a list option can't also be given bare
  (`--test` alone fails with "Parameter --test requires an argument"), but
  `--test .`, `--test ""` and repeated `--test A --test B` all parse.
- **SOJ big-band books** (surveyed 2026-10-16), Drive folder `Ongoing/SOJ`:
  - Libraries: `MasterList` (161 entries) and `Repertoire`, a renamed
    25-title subset that `SOJ/tunery.yaml` points to. Most titles are one
    full-score PDF (`Title-Arranger.pdf`); some are part folders
    (`Hay Burner-Nestico/Bass.pdf`, `Gato Gordo-Burger/Gato - Bass.pdf`).
  - Hand layouts per instrument: 26-07-09 bass and piano, 26-08-07 bass,
    26-09-12 bass (the latest made from `MasterList` plus `scores.json`).
    About two thirds of their charts are page ranges inside a full score,
    which no name reveals.
  - `apps/SOJ/data/scores.json` (32 titles; parts `bass`, `piano`): values
    are `page` (4), `[page, length]` (31), `{file}` (11) and
    `{list: [26, 25]}` (1, pages in that order). `_path` is relative to
    `MasterList`: 30/32 resolve. "Ain't That a Kick in the Head" differs
    only by apostrophe (`’` in `MasterList`); `Love is Here to Stay (Bb)-
    Wolpe.pdf` has no file.
  - Matching titles fuzzily (ratio ≥ 88), the index reproduces 26-07-09 bass
    15/15, 26-08-07 bass 21/21, 26-07-09 piano 14/15 (Walking My Baby Back
    Home: layout pages 31–34, index `[33, 4]`), and 26-09-12 bass 15/25 (10
    titles have no index entry).
  - The existing rendered books' page counts equal what their layouts imply:
    26-07-09 bass 28, piano 44; 26-08-07 bass 44; 26-09-12 bass 51.
- **Index titles in `MasterList`** (2026-10-16, through
  `apps/SOJ/cache/GDrive`, current code):
  - all 32 `scores.json` titles find candidates;
  - all 11 `{file}` parts are already found by name with the right part
    (Gato, Hooked on a Feeling `Keys` → Piano, I Feel Good's Finale names,
    Mack's `bss`);
  - only "I've Got You Under My Skin" finds several jumbo files (`- Ebm.pdf`
    and its `- others` folder);
  - a split part appears once in the hand layouts: 26-06-07's Come Dance
    With Me is a section with `Page 1` (p. 26) and `Page 2` (p. 25);
  - 26 of 32 entries keep page parts after dropping `{file}` parts.
- **Config discovery today** resolves the layout directory physically. From
  `JBM/GDrive/Shows/26-10-21` (`GDrive` → Google Drive), after the author
  removed the Drive-side `tunery.yaml`, it finds no config and renders
  nothing from Vault. `apps/JBM/tunery.yaml` (rb-index, then `./Vault`) is
  reachable only through the logical tree. Its `./Vault` record has
  `match: fuzzy` (author, 2026-10-14).

## Builds

### LIB4: Title-only combo layouts render the requested part from a nested folder library

Status: implemented 2026-10-15 (commits LIB4.*; LIB4.6–LIB4.8, verbosity
levels and `--part Rhythm` acceptance added by the author after approval on
2026-10-14; LIB4.9 added 2026-10-15)

Review 2026-10-15:
- Suite: 128 passed.
- Live run (with LIB4.9): 15 Vault titles, 6 rb-index, `not found "It Don't
  Mean Thing"`; nested bookmarks for Mack the Knife and Straighten Up only;
  `-vv` lists candidates and rules.
- Extra: a `tie for "Steamroller Blues"` warning, although the tie-break
  picks the right file (see LIB5).
- Before LIB4.9, Si Tu Vois Ma Mare took the Alto Sax chart (fuzzy match
  consumed the whole stem). Moondance added `Tenro Sax part` as its concert
  chart. Both are fixed and covered by tests.

**Improvement:** run from `apps/JBM`,
`tunery render GDrive/Shows/26-10-21/26-10-21.yaml --part Rhythm` renders the
rhythm-section book: 15 titles from the JBM Vault, each with its rhythm
parts chosen by the chains above. Six come from rb-index, and the run warns
about ties and missing parts. Today the same layout
finds no config, or, with the old Drive config, takes everything from real
books.

**Prerequisite (done):** the keys feature is committed (`3f53c4d`).

**Patches:**

- **LIB4.1** Discover configuration from the base directory. Add `--base`
  to `render` and `lookup`. Default it as in Configuration: relative layout
  path → logical working directory; absolute layout path → layout directory,
  walking up to the root. Search `tunery.yaml` and `.tunery.yaml`; both in
  one directory is an error. Tests: each file name, both present, relative
  and absolute layout paths through a symlinked directory, explicit
  `--base`, notice when none.
- **LIB4.2** Interpret item names (`tunery/parts.py`): normalization,
  boundaries, title match with remainder, combo-profile rules right to left,
  vocabulary, dual-instrument scores, part/variation/free tags. Unit tests on the combo names in
  Facts and every 26-10-02 Vault file name.
- **LIB4.3** Discover directory libraries recursively: symlinks without
  cycles, `_archive` excluded, `_.*` transparent, name texts = stem +
  innermost title folder. Existing flat-library tests pass unchanged.
- **LIB4.4** Select parts: `render --part`, substitution chains, library
  priority, tie warnings, `-k` as group without substitution,
  `<stem>.<Part>.pdf` output, indexed libraries as concert sources. Fixture:
  a temp tree with the Vault file names of the 15 titles. It asserts each
  pick from Facts and the tie warnings.
- **LIB4.5** Verbosity: `-v`/`--verbose 1` reports files used and
  substitutions; `-vv`/`--verbose 2` adds candidates and fired rules. Tests:
  each level's lines on the LIB4.4 fixture; default output unchanged apart
  from warnings.
- **LIB4.6** Combine parts: one output per `--part` option; a
  comma-separated value combines its parts in one file. Same-file parts are
  included once, missing parts are reported, and a title with several files
  gets nested part bookmarks (one file stays flat). Tests: two `--part`
  options → two outputs; `--part Bass,Piano` on a two-part title → nested
  bookmarks Bass, Piano; on a concert-only title → one flat chart.
- **LIB4.7** Rhythm section: `Rhythm`/`RS`/`Rhythm Section` expand to
  `Piano,Guitar,Bass,Drums`, with the Guitar and Drums chains. Fixture: the
  26-09-15 part files from `_Petes`/`SheetMusic`. It asserts the 23 picks in
  Facts (22 as in `26-09-15.rs.yaml`, Norwegian Wood plus Piano:Guitar).
- **LIB4.8** Score key per chart: layout `key:` filters candidates by score
  key (from folder or stem), falling back with a warning. Tests: the
  `Aint Misbehavin` fixture with `key: F` picks `Concert F`, `key: Bb` picks
  `Concert Bb`, `key: E` warns and falls back.
- **LIB4.9** Fine-tune from live runs (author, reviewed 2026-10-15):
  - fuzzy titles match word prefixes, keeping the part (Si Tu Vois Ma Mare);
  - after a hard boundary, unknown words are free tags;
  - multi-word instruments and parenthesized instruments are not duals;
  - Bass Trombone and an unspecified Saxophone are added to the vocabulary;
  - a title's parts all come from one library;
  - `-o` ending in the part label is not suffixed again.

**Acceptance:**

- *Scripted (pytest, the existing gate):* all existing tests, including the
  keys tests, pass unchanged; the LIB4.1–LIB4.9 tests pass.
- *Live (executor):* from `/Users/sasha/personal/wip/apps/JBM`, run
  `tunery render GDrive/Shows/26-10-21/26-10-21.yaml --part Rhythm -v -o
  <scratchpad>/26-10-21.Rhythm.pdf`. Observe:
  - 15 titles found under `Vault/` with the files listed in Facts
    (`--part Rhythm`), including `Ain't MisBehavin'` from `Concert F`;
  - nested bookmarks for `Mack the Knife` (Piano, Guitar, Bass) and
    `Straighten Up and Fly Right` (Piano, Guitar), flat bookmarks elsewhere;
  - 6 found in rb-index books;
  - `not found "It Don't Mean Thing"`;
  - tie warnings for `Mo Better Blues`, `All of Me` and `Girl From Ipanema`;
  - missing-part reports (e.g. Drums), and with `-v` the substitutions;
  - the PDF's bookmark tree matches the above.
  Then run once more with `-vv` and observe candidates and rules per title.

**Author action required at completion:**

- Review the rendered PDF musically.

### LIB5: Three Crosswinds books from one title list

Status: implemented 2026-10-15 (commits LIB5.*; follow-up `2f005ba` adds
`build.py gate` and docs outside the plan)

**Improvement:** one title-only layout renders the Crosswinds show's three
books (rhythm section, Norm's tenor book and the full band book) from
SheetMusic. Every title either takes the charts of the hand-made layout or
prints a warning naming the competing source. Today:
- the rhythm-section book mixes Poinciana's arrangements, picks the
  iRealPro Creepin' and mislabels parts;
- the tenor book matches 14 of 23 titles and the full book 11 of 23;
- no layout can name books or map a title's parts.

Runs from a parent folder no longer scan an unrelated tree, and a chart can
point `file:` at its score folder.

**Patches:**

Fixtures recreate library file names (copied from the libraries in Facts)
as blank PDFs under pytest's temporary directory, as the LIB4 tests do. No
test depends on absolute paths or the author's drives; only the live check
reads the real libraries.

- [x] **LIB5.1** Lint gate passes. `AGENTS.md` (with `CLAUDE.md` →
  `@AGENTS.md`) names the gate: `uv run ruff check tunery tests` plus
  `uv run pytest -q`. Ruff runs its defaults plus `PLC0415` (imports at
  module top). The author committed the setup and `ruff --fix`'s 15
  corrections (`a1c0f99`). The remaining 33 are fixed by hand: 21 inline imports, 5
  unused `vault` names in tests, `= []` CLI defaults, nested `if`s, `Self`
  for `Composer.__enter__`, the positional-only `__context`, the unused
  `best_cand`, and the dead if/else at `parts.py` (`parts = [part]` either
  way). Test: the gate passes.
- [x] **LIB5.2** Name child bookmarks by the file's part label (`Rhythm`,
  `Piano/Guitar`, `Concert`), not the requested part. Test: Mo' Better
  Blues, Norwegian Wood and Poinciana bookmark trees.
- [x] **LIB5.3** Name fixes:
  - `tenor (horn 2)` is Tenor Sax as Horn 2;
  - `v2`, `v3`, … are version tags (variations);
  - `Copy of` and other noise words add to the extra-word count.

  Tests: `The Cape Verdean Blues - tenor (horn 2)` matches Tenor Sax and
  Horn 2; `Tenor Sax.v2` has variation `v2`; `Copy of Shaker Song - Tenor`
  counts one more extra word than `Shaker Song - Tenor`.
- [x] **LIB5.4** Rhythm section inside comma lists:
  `Concert,Trumpet,Tenor Sax,Rhythm` expands `Rhythm` to its four members.
  Test: Mo' Better Blues gets Trumpet, Tenor Sax, Rhythm, Bass, Drums.
- [x] **LIB5.5** Ranks, sources, choice per part and warnings (Selection). This
  replaces LIB4's per-step ties.
  Tests:
  - Poinciana: Concert + Bass from `_Petes` for the rhythm section, plus
    `Poinciana.Bb.Emaj` for Trumpet;
  - Creepin' → `Creepin.pdf`;
  - `Take Five Bb Tenor` vs `Take Five.Bb` → same rank, one warning;
  - I Can't Help It with `SheetMusic/I Cant Help It` → a warning naming
    both folders;
  - one warning per title at most.
- [x] **LIB5.6** Named parts and per-chart choices in the layout:
  - a `part:` item in `config:` defines named parts (`--part norm` →
    `<stem>.norm.pdf`);
  - a chart's `part:` maps requested parts to others;
  - a chart's `tag:` (one or several) prefers candidates carrying them;
  - a title folder's `(in F)` gives its score key, as `- Concert F` does,
    so `key: F` finds `Norwegian Wood (in F)`.

  Tests: Norwegian Wood `{Tenor Sax: Horn 1}` under `norm`; Funk in Deep
  Freeze `{Tenor Sax: Flute}`; I Love Lucy `tag: v2` → `Tenor Sax.v2`;
  an unknown name in `--part` is a visible error.
- [x] **LIB5.7** `--explicit-bookmark-hierarchy` prefixes part bookmarks with
  `_ `. Test: `_ Bass` under Mo' Better Blues.
- [x] **LIB5.8** File-system view (Concepts, Configuration):
  - the base is `--base` or `$PWD`, and relative command-line paths join it;
  - discovery starts at the layout's directory and walks up lexically to
    the root, back through links;
  - with no config, the library is the layout's directory;
  - messages show view paths, and `-v` prints the config used.

  This replaces LIB4.1's "relative path → start at the working directory".
  Tests:
  - `proj/tunery.yaml` and a link `proj/Link` → a temp folder holding
    `Shows/x/x.yaml`; run from `proj` and from its parent, both find
    `proj/tunery.yaml`;
  - `--base` makes a relative layout path resolve against it;
  - with no config, only the layout folder is searched;
  - `found` shows `Link/…`, not the resolved path.
- [x] **LIB5.9** Output names ending in an alias of the requested part are not
  suffixed again. Test: `-o x.rs.pdf --part Rhythm` → `x.rs.pdf`.
- [x] **LIB5.10** Score folder per chart: `file:` naming a folder picks the
  title's parts among the PDFs in it. Tests: a folder whose names don't
  match the title gives the right part; a missing folder is a visible error
  naming the path.
- [x] **LIB5.11** Crosswinds 26-09-15 test case. As built: data in
  `tests/data/` (library names, the author's layout plus the planned Funk
  mapping, and the hand books' charts). The test records each designed
  difference, and asserts the `Bb Parts` choices are warned.
  - `tests/data/crosswinds-26-09-15.txt` lists, relative to SheetMusic,
    every PDF whose folder or name matches one of the 23 titles: names only,
    taken once from the author's drive.
  - The test rebuilds that tree as blank PDFs, then renders the three books
    from the layout in "Author action" below:
    - `rs` must equal `26-09-15.rs.yaml`, plus Norwegian Wood's
      Piano:Guitar;
    - `norm` must equal `26-09-15.norm.yaml`;
    - `full` must equal `26-09-15.yaml`;
    - every expected difference listed under Acceptance comes with its
      warning.

- [x] **LIB5.12** Break ties at one rank by the fewest substitution steps
  (Choice per part, step 2), before the top source. Found by acceptance; the
  author asked for the step-count form and for it to be checked in as
  LIB5.12 (2026-10-15). Tests: Cape Verdean keeps `tenor (horn 2)` for
  Tenor Sax; Alto Sax 2 takes Alto Sax over Eb.

**Acceptance:**

- *Scripted (the gate in `AGENTS.md`: ruff + pytest):* all existing tests pass; the LIB5.2–LIB5.11 tests
  pass.
- *Live (executor)*, from `/Users/sasha/personal/crosswinds`, with the
  author's layout `GDrive/Shows/26-09-15/26-09-15.1.yaml`: run `tunery
  render GDrive/Shows/26-09-15/26-09-15.1.yaml --part rs --part norm --part
  full --explicit-bookmark-hierarchy -v -o <scratchpad>/26-09-15.1.pdf`.
  Compare each book's `-v` part lines with its hand layout, title by title:
  - `-v` names `GDrive/Shows/tunery.yaml`; three files are written:
    `.rs.pdf`, `.norm.pdf`, `.full.pdf`;
  - `rs`: all 23 titles match, plus Norwegian Wood's Piano:Guitar;
  - `norm`: all match, except four titles that take a `Bb Parts` chart
    (`Take Five`, `Ceora`, `Mercy, Mercy, Mercy`, `Red Clay`), each with
    a warning;
  - `full`: all match, except:
    - the same four titles also take their `Bb Parts` chart;
    - The Cape Verdean Blues adds SheetMusic's `.pdf` and `.Bb`;
    - Norwegian Wood adds SheetMusic's `Norwegian Wood.pdf`;
    - I Can't Help It adds `I Cant Help It.pdf`.

    Each comes with a warning naming the other folder.
  - With `Bb Parts` moved out of SheetMusic, only the last three
    differences remain.
- *Live regression*:
  - re-run the LIB4 JBM check: same picks;
  - from `/Users/sasha/personal/wip`, run `tunery render
    apps/JBM/GDrive/Shows/26-10-21/26-10-21.yaml --part Rhythm -vv`: it
    must use `apps/JBM/tunery.yaml` and list no candidate outside Vault and
    the real books.

The author removed `Shows/26-09-15/tunery.yaml` and put rb-index first in
`Shows/tunery.yaml` (verified 2026-10-15). Norwegian Wood's Piano:Guitar is
expected: `26-09-15.rs.yaml` missed it (author, 2026-10-15).

**Author action required before the live check:**

- Save `GDrive/Shows/26-09-15/26-09-15.1.yaml`: the titles of
  `26-09-15.1.rs.yaml`, plus:
  ```yaml
  - config:
      - part:
          full: Concert, Trumpet, Tenor Sax, Rhythm
          norm: Tenor Sax
  ```
  and on the charts:
  - `Norwegian Wood`: `part: {Tenor Sax: Horn 1, Trumpet: Horn 2}`;
  - `Funk in Deep Freeze`: `part: {Tenor Sax: Flute}`;
  - `I Love Lucy`: `tag: v2`.
- After this build (author, 2026-10-15): move the redundant charts
  (`SheetMusic/Bb Parts`, `SheetMusic/I Cant Help It`) out of SheetMusic.
  The live check runs before that, with them in place.

**Acceptance run 2026-10-15** on tree `b14fd59` (LIB5.11) plus one
uncommitted fix (below). Environment: Python 3.12.12, ruff 0.17.1, host L25,
macOS 26.6.2. Logs: session scratchpad `acceptance-LIB5/`.

- Gate: `uv run ruff check tunery tests` clean; `uv run pytest -q` 158
  passed (157 before the fix).
- Live Crosswinds: `tunery render GDrive/Shows/26-09-15/26-09-15.1.yaml
  --part rs --part norm --part full --explicit-bookmark-hierarchy -v`, run
  from `/Users/sasha/personal/crosswinds`, exit 0.
  - `-v` names `/Users/sasha/personal/crosswinds/GDrive/Shows/tunery.yaml`.
  - Three books: `26-09-15.1.RS.pdf` (the `RS` label, per Output; the text
    above said `.rs.pdf`), `.norm.pdf`, `.full.pdf`. 23 titles each, none
    missing.
  - rs: 22/23 match, plus Norwegian Wood's Piano:Guitar, as expected.
  - norm: 18/23 match. The four `Bb Parts` titles differ, each warned.
    Funk in Deep Freeze takes Tenor, because the author's layout has no
    `{Tenor Sax: Flute}` mapping.
  - full: 18/23 match. The four `Bb Parts` titles add their tenor chart,
    and The Cape Verdean Blues adds SheetMusic's `.Bb` and `.pdf`; each is
    warned. Norwegian Wood matches exactly (`key: F` keeps SheetMusic's
    unkeyed concert chart out), and `SheetMusic/I Cant Help It` was already
    moved out.
  - Part bookmarks carry the `_ ` prefix.
- Live regression: JBM from `apps/JBM` gives the same 22 picks as LIB4.
  From `/Users/sasha/personal/wip`, `-vv` uses `apps/JBM/tunery.yaml` and
  lists no candidate outside Vault and the real books.
- Fix found by acceptance (uncommitted): before it, full-book Cape Verdean
  replaced `tenor (horn 2)` with SheetMusic's `.Bb`, because the folders
  tied as top source and path order chose SheetMusic. Now, at one rank, the
  fewest substitution steps win (Choice per part, step 2; author's
  generalization). Two new tests in `tests/test_sources.py` fail without it. The 26-09-15 test now
  expects Cape Verdean to keep its tenor chart.

**Open clauses, observed 2026-10-15** after the author moved `Bb Parts` and
`I Cant Help It` to `SheetMusic/_archive`:

- `Bb Parts` moved out: norm 22/23, full 22/23. Only The Cape Verdean Blues
  differs in full (it adds SheetMusic's `.Bb` and `.pdf`, warned). That is
  fewer differences than predicted.
- Funk in Deep Freeze → Flute: the author's layout now says
  `parts: {norm: Flute}`, keyed by a named part, which LIB5 doesn't support
  (planned in LIB6).
- Musical review: passed (author, 2026-10-15).

**Author action required at completion:**

- Review the three rendered books musically.
- Optional: add `part: {Tenor Sax: Flute}` to Funk in Deep Freeze in
  `26-09-15.1.yaml`. Note that it then applies to the full book too, which
  drops the Tenor chart there (the mapping is keyed by part, not by book).

### LIB6: Every lookup visible and checkable before rendering

Status: executed, pending author verification

**Improvement:** `tunery render gig.yaml --test .` shows, without writing a
PDF, the chart each title would use and every substitution it took.
- The Crosswinds concert book takes the band's Rhythm charts (or Piano,
  for I Can't Help It), not real-book pages or the Flute part.
- Norm's book takes Funk in Deep Freeze's Flute part through
  `parts: {norm: Flute}`.
- A typo in a part name stops the run with an error instead of a silent
  fallback.
- `-k` is gone, and collections reached through links stay in the
  file-system view.

**Patches:**

Each patch updates the design sections it changes (Selection, Concepts,
Configuration) and adds tests. Fixtures are blank PDFs under `tmp_path`.

- [x] **LIB6.1** Remove `-k`/`--keys`: `tunery/keys.py`, its tests, the
  `<Title>.<K>.pdf` suffix lookups in `DirectoryLibrary`, per-key output
  naming and `[Key]` prefixes. Retire `docs/todo/keys.md`. Tests: `-k` is
  rejected by the CLI; `--part Bb` still renders a Bb book.
- [x] **LIB6.2** One part-name parser in `tunery/parts.py`, used by `--part`,
  named-part definitions and chart mappings (keys and values). An unknown
  name is an error that names the value and, for a chart, its title.
  Tests: `{Tenor Sax: Fluet}`, `{Tenor Saxx: Flute}` and `full: Tenor Sx`
  are errors; known aliases (`Tenor`, `rs`) pass.
- [x] **LIB6.3** Chart mappings keyed by a named part: `parts: {norm: Flute}`
  replaces the title's parts when rendering `norm` and leaves other books
  alone; part keys (`Tenor Sax: Horn 1`) apply in every book as today.
  Test: Funk in Deep Freeze is Flute in `norm` and Tenor in `full`.
- [x] **LIB6.4** Concert = Lead Sheet = untagged = a Rhythm chart: one rank, in
  that step order. Then Melody, then Piano/Guitar, then Vocal (author,
  2026-10-15). Flute no longer substitutes for Concert. For Piano, Guitar,
  Bass and Drums the Rhythm chart stays ahead of Concert. Requesting `Rhythm` still means Piano,
  Guitar, Bass, Drums.
  Tests (Facts names):
  - Funk's concert chart is `- Rhythm Section`;
  - Recado takes `- Rhythm` before a real book;
  - I Can't Help It's concert chart is `- Piano`;
  - Cape Verdean's Piano part is `- rhythm`, not the concert chart;
  - Girl From Ipanema still takes `-Voice`.
- [x] **LIB6.5** Substitutions as default-level warnings, one line per title in
  the per-title warning (`Heartaches: Piano, Guitar, Bass, Drums →
  Concert`). `-v` then adds file names only. Tests: the line appears without
  `-v`, once per title; no line when nothing was substituted.
- [x] **LIB6.6** One path resolver for configs, collections and layouts. A
  collection reached through a folder link resolves lexically, like a
  project config; a symlinked collection file still resolves against its
  real location. Test: the review's `Link/collection.yaml` case shows
  `Link/scores/Tune.pdf`.
- [x] **LIB6.7** `render --test TITLE` (repeatable): resolve the named titles,
  print `-vv`-level results, write no PDF. `.` or `""` means every title
  (author, 2026-10-15). A TITLE not in the layout is an error. Tests: no
  file is written; `--test A --test B` prints only A and B; `--test .` and
  `--test ""` print every title; works with several `--part` options.

**Acceptance:**

- *Scripted:* `uv run build.py gate` passes, including the LIB6 tests and
  the LIB5 Crosswinds test updated for LIB6.3 and LIB6.4.
- *Live (executor)*, from `/Users/sasha/personal/crosswinds`, no PDF
  written:
  - `tunery render GDrive/Shows/26-09-15/26-09-15.1.yaml --test .`: the
    concert book's picks match Facts ("Concert = Lead Sheet = …"), Poinciana
    takes `_Petes`' concert chart, and each substitution is warned.
  - `--test . --part norm`: all 23 titles match `26-09-15.norm.yaml`.
  - `--test . --part full`: 20/23 match `26-09-15.yaml`:
    - The Cape Verdean Blues adds SheetMusic's `.Bb` and `.pdf` (warned);
    - Funk in Deep Freeze has no Flute, because its Concert is the Rhythm
      Section chart (LIB6.4);
    - Poinciana has no `Poinciana.Bb.Emaj.pdf`, which the author archived
      with `SheetMusic/Poinciana`; its Trumpet and Tenor take the `_Petes`
      concert chart (warned as substitutions, LIB6.5).
  - `--test . --part rs`: 22/23 match `26-09-15.rs.yaml`, plus Norwegian
    Wood's Piano:Guitar.
  - `--test Poinciana` prints only Poinciana.
- *Live regression*: from `/Users/sasha/personal/wip/apps/JBM`,
  `tunery render GDrive/Shows/26-10-21/26-10-21.yaml --part Rhythm --test .`
  gives the LIB4 picks (22 titles); `-k Bb` is rejected.
- *Open if not observed:* name each clause not run, with what it takes.

**Acceptance run 2026-10-16** on tree `ee2cb34` (clean). Environment:
Python 3.12.12, ruff 0.17.1, host L25, macOS 26.6.2. Logs: scratchpad
`acceptance-LIB6/` (`lib6-concert/norm/full/rs/poinciana/jbm.log`).

- Gate: `uv run ruff check tunery tests build.py` clean; `uv run pytest -q`
  170 passed. The LIB5 Crosswinds test needed no updates for LIB6.3/6.4.
- Live Crosswinds, from `/Users/sasha/personal/crosswinds`, exit 0, no PDF
  written in any `--test` run:
  - concert `--test .`: Recado takes `- Rhythm` over the real book;
    Poinciana takes `_Petes`' concert chart; substitutions warned per title
    (e.g. `Bolivia: Trumpet, Tenor Sax → Bb`, `Poinciana: Trumpet, Tenor
    Sax, Piano, Guitar, Drums → Concert`); every pick lists candidates
    and rules.
  - norm: 22/23 match `26-09-15.norm.yaml`. The hand `Mo's Better Blues`
    is the layout's `Mo' Better Blues` (spelling) and matches its Tenor
    Sax chart. Poinciana differs: the hand `Bb.Emaj` is archived with
    `SheetMusic/Poinciana`, so Tenor Sax takes `_Petes`' concert chart
    (warned as a substitution).
  - full: Cape Verdean adds `.Bb` and `.pdf` (warned, multi-folder);
    Funk has no Flute (Concert is the Rhythm Section chart, LIB6.4);
    Poinciana has no `Bb.Emaj` (archived), Trumpet and Tenor take the
    `_Petes` concert chart (warned as substitutions).
  - rs: hand file set 29/29 picked, plus Norwegian Wood's Piano:Guitar,
    as designed.
  - `--test Poinciana` prints only Poinciana.
- Live regression: JBM `--part Rhythm --test .` gives the LIB4 picks (21
  found under Vault (15) and books (6), plus the `It Don't Mean Thing`
  typo); `-k Bb` is rejected as an unknown option.
- No code fixes were needed; nothing is left uncommitted.

**Author action required at completion:**

- Render the three Crosswinds books and the JBM book, and review them
  musically.

### LIB7: SOJ per-instrument books from a title-only setlist

Status: planned, approved 2026-10-16

**Improvement:** a big-band setlist of titles alone renders the bass or piano
book. Page ranges come from an index inside the library definition
(`MasterList/tunery.yaml`), part files from their names. Today these books
are hand-written per instrument and gig. A jumbo or combined-set PDF never stands
in for a missing part; the part is reported missing instead.

**Patches:**

Each patch updates the design sections it changes and adds tests (blank
PDFs under `tmp_path`, names copied from the libraries).

- [x] **LIB7.1** Library `type: combo | big band` on library records (a
  collection reference passes it on like `match`).
  - A big-band library holds the score formats of `apps/SOJ/docs/SPEC.md`
    ("Score Format"; the ZIPs are unpacked into folders):
    - **parts**, one part per file (`Hay Burner-Nestico/Bass.pdf`): found
      by name, as today;
    - **jumbo**, every part in sequence in one file, named for no part
      (`Bandstand Boogie-Lowden.pdf`);
    - **combined set**, a few related parts in sequence, named for their
      group (`Rhythm Parts.pdf`).

    A jumbo or combined-set file is never reached by a chain, so it never
    stands in for a part; its parts come from the index (LIB7.2).
  - **pages** (one page of a part per file, `Bone 2 Pg 2`) and
    **multi-staff** (`Tenors n Bari`) stay in the backlog: none of the four
    SOJ books uses them.
  - A conductor score (`conductor`, `score`, `condensed score`) is a part of
    its own (`Conductor`), all instruments on one page: recognized and
    listed, never substituted for another part (author, 2026-10-16).
  - Arrangers need no rule: `-Lowden` is a free tag, which `tag: Lowden`
    already selects (author, 2026-10-16).

  Tests:
  - `Bandstand Boogie-Lowden.pdf` is jumbo and `Rhythm Parts.pdf` a combined
    set in a big-band library;
  - `--part Bass` on it alone reports `missing Bass`;
  - `Hay Burner-Nestico/Conductor.pdf` is Conductor and never answers
    `--part Piano`;
  - in a combo library the same file stays Concert.
- [x] **LIB7.2** Part index in the library definition (author, 2026-10-16):
  ```yaml
  # GDrive/MasterList/tunery.yaml, included by GDrive/tunery.yaml
  - library: ./
    type: big band
    match: fuzzy
    index:
      - title: Bandstand Boogie
        parts: {bass: 15, piano: [17, 3]}
      - title: I've Got You Under My Skin
        file: I've Got You Under My Skin - Ebm.pdf
        parts: {bass: [27, 2]}
  ```
  - An entry names the title. Its file is the title's jumbo file in the
    library, found like any title. `file:` (relative to the library) names
    the file explicitly, however its name reads. It is needed when the title
    finds several jumbo files ("I've Got You Under My Skin"), for a combined
    set, or for a file whose name reads as something else
    (`Love Is Here To Stay.Low Key.Bb.pdf`, where `Low Key` is a tag for a
    low-voiced singer; author, 2026-10-16).
  - Part values are `page`, `[page, length]`, or a list of ranges
    `[[26, 1], [25, 1]]` for pages out of order (author, 2026-10-16).
  - Part names go through the part parser (LIB6.2), so an unknown name is an
    error. Part files found by name need no entry (Gato Gordo, Hay Burner, …).
  - Index parts are candidates of the title at the part's own step.
  - An entry whose title finds no jumbo file, or several without `file:`,
    gets a warning naming the title, and its parts count as missing.

  Tests:
  - the value shapes;
  - an entry without `file:`;
  - the `file:` case;
  - no file and several files warn;
  - an index part and a part file named for the same title coexist.
- [x] **LIB7.3** A part given as a list of ranges renders its ranges in list
  order under one bookmark for the title. The composer adds continuation
  ranges without their own bookmark. It is a one-off (Come Dance With Me's
  pages are out of order), so the simplest rendering will do (author,
  2026-10-16). Test: Come Dance With Me's bass is pages 26 then 25, with one
  bookmark.
- [x] **LIB7.4** SOJ test case.
  - `tests/data/` gets the `MasterList` names for the four books' titles,
    the converted index, and the hand books' charts (file and pages per
    title).
  - The test renders each book from its titles. It asserts:
    - 26-07-09 bass 15/15 and 26-08-07 bass 21/21 as the hand layouts;
    - 26-07-09 piano 15/15 as the index (Walking My Baby Back Home is pages
      31–35, unlike the hand layout's 31–34);
    - 26-09-12 bass 15/25, with `missing Bass` for the 10 titles without an
      entry.

**One-off data conversion (declared, author-applied):** the executor
converts `apps/SOJ/data/scores.json` into the index above. That gives 26
entries; the 6 whose parts are all found by name are dropped, and `_path` is
dropped. `file:` is kept for "I've Got You Under My Skin" and set to
`Love Is Here To Stay.Low Key.Bb.pdf` for Love is Here to Stay. Walking My
Baby Back Home's piano becomes `[31, 5]` (author, 2026-10-16: pages 31–35). The
result goes to the scratchpad as `MasterList.tunery.yaml`, for the author to
review and place in Drive. The executor writes nothing to Drive.

**Acceptance:**

- *Scripted:* `uv run build.py gate` passes, including the LIB7 tests.
- *Live (executor)*, from `/Users/sasha/personal/wip/apps/SOJ/cache`. Use a
  scratchpad copy of the converted index whose `library:` is the absolute
  `GDrive/MasterList` path, plus, per book, a scratchpad layout of the hand
  layout's titles and sections whose `config:` includes that copy. Render
  with `--part Bass` (or `Piano`) and observe:
  - 26-07-09 bass and 26-08-07 bass: every chart's file and pages equal the
    hand layout's, and the books have 28 and 44 pages, like the existing
    `26-07-09.bass.pdf` and `26-08-07.bass.pdf`;
  - 26-07-09 piano: 45 pages (the existing book has 44); only Walking My
    Baby Back Home differs from the hand layout (pages 31–35, not 31–34);
  - 26-09-12 bass: 15 titles as the hand layout, and `missing Bass` for the
    10 without an index entry;
  - `--test .` on the 26-08-07 layout lists each title's index part as a
    candidate.
- *Live regression:* the LIB6 Crosswinds and JBM `--test .` runs give the
  same picks (combo libraries).
- *Open if not observed:* name each clause not run, with what it takes.

**Author action required at completion:**

- Review the converted `MasterList.tunery.yaml` and place it as
  `GDrive/MasterList/tunery.yaml`. Point `GDrive/tunery.yaml` at it
  (`- library: ./MasterList/tunery.yaml`) instead of `Repertoire`.
- Optional: add index entries for the 10 titles of 26-09-12 that lack one.
- Review the rendered books musically.

## Backlog

- Group folders (`Bb Parts`) add their part to everything below.
- Sibling-prefix grouping for name texts (rules in Facts).
- Rest of the big-band profile (rules marked big band above): trailing key,
  big-band instruments and codes, arrangement version (`type`, Score
  default and arranger: LIB7).
- Durable named parts in the config (layout-level names: LIB5).
- Variation selection by arranger (score key: LIB4.8; tags: LIB5).
- Forced per-item metadata in library records; later, groups selected by
  pattern.
- Multi-file page splits concatenated as one part (`P1`/`P2`, `_pg 2`).
- Library exclusion list (`.gitignore` analog) beyond `_archive`.
- Inspection command listing discovered titles, parts and free tags.
- `lookup --part`.
- `path:` hint per chart: prefer candidates whose path contains a fragment
  or pattern (`file:` covers an exact folder).
- Big-band best-effort: acronym and camelCase titles (`fmttm…`, `gd…`,
  `FlyingHomeRHYTHM`).
- Per-patch interaction tests (new path vs. keys, collections, `file:`,
  mappings, links): postponed to /onereview.
- Full score: every part of every title in one file.
