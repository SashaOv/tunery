"""Interpret combo library file/folder names into parts and variations.

Combo profile only (big-band rules disabled). See docs/todo/library-based.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz


COMBO_DEFAULT_PART = "Concert"

# Instrument canonical -> (aliases, group). Aliases in normalized space form.
_INSTRUMENTS: list[tuple[str, list[str], str | None]] = [
    ("Soprano Sax", ["soprano sax", "soprano"], "Bb"),
    ("Alto Sax", ["alto saxophone", "alto sax", "alto", "alt", "as"], "Eb"),
    ("Tenor Sax", ["tenor saxophone", "tenor sax", "tenorsax", "tenor", "ten", "ts"], "Bb"),
    ("Bari Sax", ["bari sax", "baritone sax", "bari", "baritone", "bsx"], "Eb"),
    # Unspecified saxophone (e.g. "Tenro Sax part", "John Mayer-Saxophone"):
    # a sax part of unknown key, so it never passes for the concert chart.
    ("Saxophone", ["saxophone", "sax"], None),
    ("Bass Clarinet", ["bass clarinet", "bass clr"], "Bb"),
    ("Clarinet", ["clarinet in bb", "clarinet"], "Bb"),
    ("Trumpet", ["trumpet in bb", "trumpet", "tpt", "tp"], "Bb"),
    ("Flugelhorn", ["flugelhorn", "flugel"], "Bb"),
    ("Trombone", ["trombone", "bone", "tbn"], "Bass Clef"),
    ("Bass Trombone", ["bass trombone"], "Bass Clef"),
    ("Flute", ["flute"], "Concert"),
    ("Vocal", ["vocal", "voice", "vx"], "Concert"),
    ("Piano", ["electric piano", "piano", "pno", "keys"], "Rhythm"),
    ("Guitar", ["electric guitar", "guitar", "gtr", "guit"], "Rhythm"),
    ("Bass", ["electric bass", "bass", "bss"], "Rhythm"),
    ("Drums", ["drums", "drm"], "Rhythm"),
    ("Percussion", ["percussion"], "Rhythm"),
    ("Horn", ["horn"], None),
]

# Alias lookup: normalized alias -> (canonical, group)
_ALIAS_MAP: dict[str, tuple[str, str | None]] = {}
for _canon, _aliases, _group in _INSTRUMENTS:
    for _a in _aliases:
        _ALIAS_MAP[_a] = (_canon, _group)
        _ALIAS_MAP[_a.replace(" ", "")] = (_canon, _group)

# Squashed alias forms (spaces removed) for single-instrument detection.
# Used to keep multi-word instruments ("bass clarinet") from being read as
# duals ("Bass"+"Clarinet").
_SQUASHED_ALIAS_SET: set[str] = {a.replace(" ", "") for a in _ALIAS_MAP}

_GROUPS: dict[str, str] = {
    "bb": "Bb",
    "eb": "Eb",
    "concert": "Concert",
    "c instruments": "Concert",
    "c instr": "Concert",
    "melody": "Melody",
    "rhythm": "Rhythm",
    "rhythm section": "Rhythm",
    "rs": "Rhythm",
}

_CONCERT_FALLBACK = ["Concert", "Lead Sheet", "Vocal", "Melody", "Flute"]

_INSTRUMENT_GROUPS: dict[str, str | None] = {
    canon: group for canon, _, group in _INSTRUMENTS
}

_KINDS: dict[str, str] = {
    "lead sheet": "Lead Sheet",
    "lead sheets": "Lead Sheet",
    "score": "Score",
    "lyrics": "Lyrics",
    "solo": "Solo",
    "solos": "Solo",
    "changes": "Changes",
    "structure": "Structure",
    "harmony": "Harmony",
}

_NOISE_WORDS = {
    "original",
    "rescan",
    "v2",
    "v3",
    "org",
    "irealpro",
    "ireal",
    "part",
    "parts",
    "in",
    "with",
    "the",
    "n",
    "and",
    "&",
    "a",
    "an",
    "of",
    "for",
    "no",
}

_VALID_KEYS = {"c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"}
_KEY_CANONICAL = {
    "c": "C",
    "db": "Db",
    "d": "D",
    "eb": "Eb",
    "e": "E",
    "f": "F",
    "gb": "Gb",
    "g": "G",
    "ab": "Ab",
    "a": "A",
    "bb": "Bb",
    "b": "B",
}


def normalize_text(text: str) -> str:
    """Lowercase, curly apostrophe to straight, drop apostrophes, punct/_ to space."""
    s = text.lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("'", "")
    s = re.sub(r"[_\W]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def squash_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def fuzzy_title_prefix(title: str, name_text: str, cutoff: float = 90.0) -> str | None:
    """Word prefix of `name_text` that fuzzily equals `title`, or None.

    Compares squashed forms; the best ratio at or above `cutoff` wins, the
    shortest prefix on equal ratios. Returns the prefix in normalized form,
    which `match_candidate` matches exactly against the same name text.
    """
    target = squash_text(title)
    words = normalize_text(clean_stem(name_text)).split(" ")
    best: int | None = None
    best_score = cutoff
    for end in range(1, len(words) + 1):
        score = fuzz.ratio(target, "".join(words[:end]))
        if score > best_score or (best is None and score >= cutoff):
            best, best_score = end, score
    return " ".join(words[:best]) if best is not None else None


def clean_stem(stem: str) -> str:
    """Unwrap Finale prefix, drop Copy prefixes and trailing (1)."""
    s = stem.strip()
    m = re.match(r"^Finale 2004a\s*-\s*\[(.*)\]\s*$", s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"^(Copy of |Copy \(\d+\) of )", "", s)
    s = re.sub(r"[\s_\-]*\(1\)\s*$", "", s)
    return s


def _parse_key_token(token: str) -> str | None:
    t = token.strip().lower().replace("♯", "#").replace("♭", "b")
    t = t.replace("#", "").replace("sharp", "").strip()
    # Handle trailing b for flat (bb -> Bb, eb, etc.)
    if t in _KEY_CANONICAL:
        return _KEY_CANONICAL[t]
    # Sharps: c#, d#, etc. map to canonical via base?
    sharp_map = {"c#": "C#", "d#": "D#", "f#": "F#", "g#": "G#", "a#": "A#"}
    if t in sharp_map:
        return sharp_map[t]
    return None


@dataclass
class InterpretedTags:
    parts: list[str]
    part: str
    score_key: str | None = None
    score_mode: str | None = None
    variations: list[str] = field(default_factory=list)
    free_tags: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)


def _tokenize_tag_text(tag_text: str) -> list[str]:
    # Split on hard boundaries: ' - ', '-', '.', ',', '(', ')', '[', ']', ':', '/', '+', '&', '|'
    # Keep words; hard separators become token boundaries (spaces already normalized).
    # normalize first, then split on spaces.
    norm = normalize_text(tag_text)
    return [t for t in norm.split(" ") if t]


def interpret_remainder(tag_text: str, allow_dual: bool = True) -> InterpretedTags:
    """Interpret tag text right to left, longest span wins, combo profile."""
    tokens = _tokenize_tag_text(tag_text)
    n = len(tokens)
    consumed = [False] * n
    rules_fired: list[str] = []
    score_key: str | None = None
    score_mode: str | None = None
    instruments_found: list[tuple[int, str, str | None]] = []  # index, canon, role?
    groups_found: list[tuple[int, str]] = []
    kinds_found: list[tuple[int, str]] = []
    dual: list[str] | None = None

    # Check dual-instrument single-segment: whole remainder joins two
    # instruments AND is not itself a single multi-word instrument alias
    # (e.g. "bass clarinet" is one instrument, not a Bass+Clarinet dual).
    # Parenthesized instruments are variations, never duals
    # (e.g. "All of Me (Vocal)-Piano" -> Piano, variation vocal).
    squashed_all = "".join(tokens)
    has_parens = "(" in tag_text or "[" in tag_text
    if tokens and allow_dual and not has_parens and squashed_all not in _SQUASHED_ALIAS_SET and _is_dual(squashed_all, tokens):
        dual_parts = _split_dual(squashed_all)
        if dual_parts:
            return InterpretedTags(
                parts=dual_parts,
                part="/".join(dual_parts),
                variations=[],
                free_tags=[],
                rules_fired=["dual-instrument"],
            )

    i = n - 1
    while i >= 0:
        if consumed[i]:
            i -= 1
            continue
        # Try spans ending at i, longest first, up to 3 words.
        matched = False
        for span in (3, 2, 1):
            start = i - span + 1
            if start < 0:
                continue
            if any(consumed[start : i + 1]):
                continue
            phrase = " ".join(tokens[start : i + 1])
            # Explicit score key: concert <key>
            if span == 2 and tokens[start] == "concert":
                key = _parse_key_token(tokens[i])
                if key:
                    score_key = key
                    rules_fired.append(f"explicit score key: {phrase}")
                    for k in range(start, i + 1):
                        consumed[k] = True
                    matched = True
                    break
            # Explicit: in <key>
            if span == 2 and tokens[start] == "in":
                key = _parse_key_token(tokens[i])
                if key:
                    score_key = key
                    rules_fired.append(f"explicit score key: {phrase}")
                    for k in range(start, i + 1):
                        consumed[k] = True
                    matched = True
                    break
            # Explicit: <key> major/minor
            if span == 2 and tokens[i] in ("major", "minor"):
                key = _parse_key_token(tokens[start])
                if key:
                    score_key = key
                    score_mode = tokens[i]
                    rules_fired.append(f"explicit score key: {phrase}")
                    for k in range(start, i + 1):
                        consumed[k] = True
                    matched = True
                    break
            # Explicit: <key>m (e.g. gm, cm) single token ending with m
            if span == 1 and len(tokens[i]) >= 2 and tokens[i].endswith("m"):
                base = tokens[i][:-1]
                key = _parse_key_token(base)
                # Avoid clashing with drum alias etc. (drm not a key base)
                if key and base in _VALID_KEYS or base in ("g", "c", "d", "a", "e", "b"):
                    score_key = key
                    score_mode = "minor"
                    rules_fired.append(f"explicit score key: {phrase}")
                    consumed[i] = True
                    matched = True
                    break
            # Page split two-word: pg N / page N
            if span == 2 and tokens[start] in ("pg", "p", "page"):
                if re.fullmatch(r"\d{1,3}", tokens[i]):
                    rules_fired.append(f"page split: {phrase}")
                    for k in range(start, i + 1):
                        consumed[k] = True
                    matched = True
                    break
            # Instrument with role: <instrument> <number> (1-2 word instrument)
            if re.fullmatch(r"\d{1,2}", tokens[i]):
                inst_phrase = " ".join(tokens[start:i])
                canon_group = _ALIAS_MAP.get(inst_phrase)
                if canon_group:
                    canon, _ = canon_group
                    instruments_found.append((start, canon, tokens[i]))
                    rules_fired.append(f"instrument role: {phrase}")
                    for k in range(start, i + 1):
                        consumed[k] = True
                    matched = True
                    break
            # Two-word instrument (e.g. alto sax, tenor sax, bass clarinet, electric guitar)
            if phrase in _ALIAS_MAP and span >= 2:
                canon, _ = _ALIAS_MAP[phrase]
                instruments_found.append((start, canon, None))
                rules_fired.append(f"instrument: {phrase}")
                for k in range(start, i + 1):
                    consumed[k] = True
                matched = True
                break
            # Group two-word (c instruments, bb instr, etc.)
            if phrase in _GROUPS and span >= 2:
                groups_found.append((start, _GROUPS[phrase]))
                rules_fired.append(f"group: {phrase}")
                for k in range(start, i + 1):
                    consumed[k] = True
                matched = True
                break
            # Kind two-word (lead sheet)
            if phrase in _KINDS and span >= 2:
                kinds_found.append((start, _KINDS[phrase]))
                rules_fired.append(f"kind: {phrase}")
                for k in range(start, i + 1):
                    consumed[k] = True
                matched = True
                break
        if matched:
            i -= 1
            continue
        # Single-token rules at position i.
        tok = tokens[i]
        # Horn role with slash: e.g. tokens "horn","1","2"? After normalize, "1/2" -> "1 2".
        # Handle instrument + number(s) below via single instrument check first.
        if tok in _ALIAS_MAP:
            canon, _ = _ALIAS_MAP[tok]
            # Check for role number immediately following? We go right-to-left,
            # so role number would be at i+1 and already consumed as noise/digit?
            # Instead detect pattern instrument + digits ahead.
            instruments_found.append((i, canon, None))
            rules_fired.append(f"instrument: {tok}")
            consumed[i] = True
            i -= 1
            continue
        if tok in _GROUPS:
            groups_found.append((i, _GROUPS[tok]))
            rules_fired.append(f"group: {tok}")
            consumed[i] = True
            i -= 1
            continue
        if tok in _KINDS:
            kinds_found.append((i, _KINDS[tok]))
            rules_fired.append(f"kind: {tok}")
            consumed[i] = True
            i -= 1
            continue
        # Page split: pg N (pg already, N next), p1, p2
        if tok in ("pg", "p", "page"):
            rules_fired.append(f"page split: {tok}")
            consumed[i] = True
            i -= 1
            continue
        if re.fullmatch(r"p\d{1,3}", tok):
            rules_fired.append(f"page split: {tok}")
            consumed[i] = True
            i -= 1
            continue
        # Noise: pure digits (track numbers, 0), v2/v3, named noise
        if re.fullmatch(r"\d{1,3}", tok) or tok in _NOISE_WORDS or re.fullmatch(r"v\d+", tok):
            # Attach role numbers to preceding instrument if adjacent.
            if re.fullmatch(r"\d{1,2}", tok) and instruments_found:
                # If previous token (i-1) is an instrument just found, make it a role.
                pass
            rules_fired.append(f"noise: {tok}")
            consumed[i] = True
            i -= 1
            continue
        # Moved article trailing the/a (from ", The" / ", A") treated as noise-like.
        if tok in ("the", "a"):
            rules_fired.append(f"moved article/noise: {tok}")
            consumed[i] = True
            i -= 1
            continue
        # Unmatched: leave as free tag candidate.
        i -= 1

    # Post-process roles: instrument followed by bare number (e.g. horn 1).
    # Numbers were consumed as noise; reattach: scan tokens left-to-right for
    # instrument + number pattern to set roles.
    roles: dict[int, str] = {}
    for idx in range(n - 1):
        if tokens[idx] in _ALIAS_MAP and re.fullmatch(r"\d", tokens[idx + 1]):
            # Find instrument entry at idx.
            for j, (pos, canon, _) in enumerate(instruments_found):
                if pos == idx:
                    instruments_found[j] = (pos, canon, tokens[idx + 1])
                    roles[pos] = tokens[idx + 1]
                    # Remove noise rule for that number if present? Keep fired list.
                    break

    free_tags = [tokens[k] for k in range(n) if not consumed[k]]

    # Part determination.
    if dual:
        parts = dual
        part = "/".join(dual)
    elif instruments_found:
        # Rightmost instrument (largest index).
        instruments_found.sort(key=lambda x: x[0])
        pos, canon, role = instruments_found[-1]
        part = f"{canon} {role}" if role else canon
        parts = [part] if "/" not in part else [part]
        # Handle Horn 1/2 slash: if tokens contain horn 1 2 pattern, keep first role.
        variations: list[str] = []
        for p, c, r in instruments_found[:-1]:
            variations.append(c.lower())
        variations_out = variations
    elif groups_found:
        groups_found.sort(key=lambda x: x[0])
        part = groups_found[-1][1]
        parts = [part]
        variations_out = []
    elif kinds_found:
        kinds_found.sort(key=lambda x: x[0])
        part = kinds_found[-1][1]
        parts = [part]
        variations_out = []
    else:
        part = COMBO_DEFAULT_PART
        parts = [part]
        variations_out = []

    if instruments_found and not dual:
        pass
    else:
        variations_out = []

    # Earlier instruments become variations (only when part is instrument).
    # Already set above for instrument case.
    if instruments_found and not dual:
        pass
    elif groups_found or kinds_found:
        variations_out = []

    return InterpretedTags(
        parts=parts,
        part=part,
        score_key=score_key,
        score_mode=score_mode,
        variations=variations_out if instruments_found else [],
        free_tags=free_tags,
        rules_fired=rules_fired,
    )


def _is_dual(squashed: str, tokens: list[str]) -> bool:
    return _split_dual(squashed) is not None


def _split_dual(squashed: str) -> list[str] | None:
    """Split a single-segment squashed token into two instrument canonicals."""
    # Build alias squashed map (spaces removed).
    alias_sq: dict[str, str] = {}
    for alias, (canon, _) in _ALIAS_MAP.items():
        sq = alias.replace(" ", "")
        # Prefer longer aliases for same squashed form (e.g. electric piano).
        if sq not in alias_sq or len(alias) > len(alias_sq[sq]):
            alias_sq[sq] = canon
    # Try every split position.
    for k in range(2, len(squashed) - 2):
        left, right = squashed[:k], squashed[k:]
        if left in alias_sq and right in alias_sq:
            return [alias_sq[left], alias_sq[right]]
    return None


def match_prefix(layout_title: str, name_text: str) -> str | None:
    """Return remainder of name_text after layout title T, or None.

    Exact squashed prefix at a boundary. Hard boundaries always allow the
    split; a soft (space) boundary allows it here (tag validation happens in
    selection via interpret_remainder).
    """
    hit = match_prefix_boundary(layout_title, name_text)
    return hit[0] if hit else None


def match_prefix_boundary(layout_title: str, name_text: str) -> tuple[str, bool, bool] | None:
    """Return (remainder, is_hard_boundary, remainder_has_parens) or None.

    Hard separators: ' - ', '-', '.', ',', '(', ')', '[', ']', ':'. Soft is a
    space. No separator (mid-word) is not a boundary. Exact match returns
    ("", True, False).
    """
    name_clean = clean_stem(name_text)
    t_sq = squash_text(layout_title)
    n_sq = squash_text(name_clean)
    if not t_sq:
        return None
    if n_sq == t_sq:
        return ("", True, False)
    if not n_sq.startswith(t_sq):
        return None
    # Walk original cleaned string counting alphanumerics to find split.
    lowered = name_clean.lower().replace("’", "'").replace("‘", "'").replace("'", "")
    target = len(t_sq)
    count = 0
    split_pos: int | None = None
    for pos, ch in enumerate(lowered):
        if ch.isalnum():
            count += 1
            if count == target:
                split_pos = pos + 1
                break
    if split_pos is None:
        return None
    rest_original = lowered[split_pos:]
    if not rest_original:
        return ("", True, False)
    # Boundary check: next char must not be alnum (mid-word).
    first = rest_original[0]
    if first.isalnum():
        return None
    hard_chars = set("-.,()[]:")
    # ' - ' contains '-', hard. Space alone is soft.
    is_hard = any(c in hard_chars for c in rest_original[:3])
    has_parens = "(" in rest_original or "[" in rest_original
    # Normalize remainder via normalize_text for downstream parsing.
    norm = normalize_text(name_clean)
    # Recompute remainder via norm split (as before) for stable output.
    sq_count = 0
    idx = 0
    for p, ch in enumerate(norm):
        if ch != " ":
            sq_count += 1
            if sq_count == target:
                idx = p + 1
                break
    else:
        return None
    rest = norm[idx:].strip()
    return (rest, is_hard, has_parens)


@dataclass
class ItemInfo:
    stem: str
    folder: str | None = None
    parts: list[str] = field(default_factory=list)
    part: str = COMBO_DEFAULT_PART
    score_key: str | None = None
    variations: list[str] = field(default_factory=list)
    free_tags: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)


def parse_filename(stem: str, folder: str | None = None) -> ItemInfo:
    """Parse a file stem (+ optional title folder) into part/score key."""
    cleaned = clean_stem(stem)
    # Score key may come from folder or stem; scan both.
    folder_keys: list[str] = []
    folder_norm = normalize_text(folder) if folder else ""
    for m in re.finditer(r"concert\s+([a-g][#b]?)", folder_norm):
        key = _parse_key_token(m.group(1))
        if key:
            folder_keys.append(key)
    interp = interpret_remainder(cleaned)
    score_key = interp.score_key or (folder_keys[0] if folder_keys else None)
    return ItemInfo(
        stem=cleaned,
        folder=folder,
        parts=interp.parts,
        part=interp.part,
        score_key=score_key,
        variations=interp.variations,
        free_tags=interp.free_tags,
        rules_fired=interp.rules_fired,
    )


def canonical_part_name(requested: str) -> str:
    """Canonicalize a --part value (case-insensitive, aliases)."""
    norm = normalize_text(requested)
    if norm in ("rs", "rhythm section", "rhythm"):
        # RS expansion is LIB4.7; for single-part selection treat as Rhythm.
        return "Rhythm"
    if norm in _ALIAS_MAP:
        canon, _ = _ALIAS_MAP[norm]
        return canon
    # Instrument + role (e.g. horn 1, alto sax 2).
    m = re.match(r"^(.*)\s+(\d{1,2})$", norm)
    if m and m.group(1) in _ALIAS_MAP:
        canon, _ = _ALIAS_MAP[m.group(1)]
        return f"{canon} {m.group(2)}"
    for key, canon in _GROUPS.items():
        if norm == key:
            return canon
    for key, canon in _KINDS.items():
        if norm == key:
            return canon
    # Title-case fallback (Bb, Eb, Concert, etc.).
    low = norm.lower()
    if low == "bb":
        return "Bb"
    if low == "eb":
        return "Eb"
    return requested.strip() or COMBO_DEFAULT_PART


def _base_instrument(part: str) -> str:
    m = re.match(r"^(.*)\s+\d{1,2}$", part)
    if m and m.group(1) in _INSTRUMENT_GROUPS:
        return m.group(1)
    return part


def _folder_score_key(folder: str | None) -> str | None:
    if not folder:
        return None
    folder_norm = normalize_text(folder)
    m = re.search(r"concert\s+([a-g][#b]?)", folder_norm)
    if m:
        return _parse_key_token(m.group(1))
    return None


@dataclass
class TitleCandidate:
    remainder: str
    parts: list[str]
    part: str
    score_key: str | None = None
    variations: list[str] = field(default_factory=list)
    free_tags: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)

    @property
    def tag_count(self) -> int:
        return (1 if self.score_key else 0) + len(self.variations) + len(self.free_tags)


def match_candidate(title: str, stem: str, folder: str | None = None) -> TitleCandidate | None:
    """Match layout title against stem or folder, interpreting remainder tags.

    Hard-boundary remainders accept free tags as variations; soft-boundary
    remainders require all-tag content (Morning Piano yes, Morning Dance no).
    """
    stem_hit = match_prefix_boundary(title, stem)
    if stem_hit is not None:
        stem_remainder, is_hard, has_parens = stem_hit
        interp = interpret_remainder(stem_remainder, allow_dual=not has_parens)
        if not (interp.free_tags and not is_hard):
            score_key = interp.score_key or _folder_score_key(folder)
            return TitleCandidate(
                remainder=stem_remainder,
                parts=interp.parts,
                part=interp.part,
                score_key=score_key,
                variations=interp.variations,
                free_tags=interp.free_tags,
                rules_fired=interp.rules_fired,
            )
        # Soft-boundary remainder with free tags: not this title via the stem
        # (e.g. "Mack_The_Knife__Bobby_Darin Score-Piano" slug words). Fall
        # through to folder matching below.
    if folder:
        folder_hit = match_prefix_boundary(title, folder)
        if folder_hit is not None:
            folder_remainder, folder_hard, folder_parens = folder_hit
            folder_interp = interpret_remainder(folder_remainder, allow_dual=not folder_parens)
            if folder_interp.free_tags and not folder_hard:
                return None
            stem_interp = interpret_remainder(clean_stem(stem))
            score_key = stem_interp.score_key or folder_interp.score_key or _folder_score_key(folder)
            return TitleCandidate(
                remainder=clean_stem(stem),
                parts=stem_interp.parts,
                part=stem_interp.part,
                score_key=score_key,
                variations=stem_interp.variations,
                free_tags=[],
                rules_fired=stem_interp.rules_fired,
            )
    return None


def chain_for_part(requested: str) -> list[str]:
    """Substitution chain for a requested part, earliest first."""
    part = canonical_part_name(requested)
    if part == "Piano":
        return ["Piano", "Rhythm", *_CONCERT_FALLBACK]
    if part == "Guitar":
        return ["Guitar", "Piano", "Rhythm", *_CONCERT_FALLBACK]
    if part == "Bass":
        return ["Bass", "Rhythm", *_CONCERT_FALLBACK]
    if part == "Drums":
        return ["Drums", "Rhythm", *_CONCERT_FALLBACK]
    if part == "Rhythm":
        return ["Rhythm", *_CONCERT_FALLBACK]
    if part in _CONCERT_FALLBACK:
        # Concert step with sub-order starting at requested kind.
        idx = _CONCERT_FALLBACK.index(part)
        return _CONCERT_FALLBACK[idx:]
    base = _base_instrument(part)
    if base in _INSTRUMENT_GROUPS:
        group = _INSTRUMENT_GROUPS[base]
        steps = [part]
        if part != base:
            steps.append(base)
        if group and group not in steps:
            # Rhythm group already covered? Keep single Rhythm entry.
            steps.append(group)
        for fallback in _CONCERT_FALLBACK:
            if fallback not in steps:
                steps.append(fallback)
        return steps
    # Group or unknown: direct then Concert fallback.
    steps = [part]
    for fallback in _CONCERT_FALLBACK:
        if fallback not in steps:
            steps.append(fallback)
    return steps
