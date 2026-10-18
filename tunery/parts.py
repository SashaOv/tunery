"""Interpret library file and folder names into parts and variations.

Rule priorities per library type live in `RULES`. See docs/SPEC.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Literal

from rapidfuzz import fuzz

COMBO_DEFAULT_PART = "Concert"
BIG_BAND_DEFAULT_PART = "Score"

LibraryType = Literal["combo", "big band"]

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
    "bass clef": "Bass Clef",
    "melody": "Melody",
    "rhythm": "Rhythm",
    "rhythm section": "Rhythm",
    "rs": "Rhythm",
}

# Concert, Lead Sheet, untagged items and Rhythm charts form one rank, in
# that step order. Concert books then fall back to Melody, Piano/Guitar (one
# rank) and Vocal; Flute no longer substitutes for Concert. Instrument parts
# fall back through the concert rank to Melody and Vocal only.
_CONCERT_RANK = ["Concert", "Lead Sheet", "Rhythm"]
_CONCERT_REST = ["Melody", "Piano", "Guitar", "Vocal"]
_FALLBACK_REST = ["Melody", "Vocal"]
_CONCERT_FALLBACK = [*_CONCERT_RANK, *_CONCERT_REST]


def _chain(*steps: str) -> list[str]:
    """Substitution steps in order, first occurrence wins."""
    return list(dict.fromkeys(steps))

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

# Noise words add to a candidate's extra-word count; fillers do not.
_NOISE_WORDS = {"original", "rescan", "org", "irealpro", "ireal"}
_FILLER_WORDS = {
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


_COPY_PREFIX = re.compile(r"^(Copy of |Copy \(\d+\) of )")


def clean_stem(stem: str) -> str:
    """Unwrap Finale prefix, drop Copy prefixes and trailing (1)."""
    s = stem.strip()
    m = re.match(r"^Finale 2004a\s*-\s*\[(.*)\]\s*$", s)
    if m:
        s = m.group(1).strip()
    s = _COPY_PREFIX.sub("", s)
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


def normalize_score_key(key: str) -> str:
    """Canonicalize a chart `key:` value (`bb` -> `Bb`)."""
    token = key.strip().replace("♯", "#").replace("♭", "b")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"(?i)(?<=[A-Ga-g])-?sharp", "#", token)
    token = re.sub(r"(?i)(?<=[A-Ga-g])-?flat", "b", token)
    if token.upper() == "CONCERT":
        return "C"
    upper = token.upper()
    aliases = {
        "C": "C",
        "DB": "Db",
        "C#": "C#",
        "CSHARP": "C#",
        "D": "D",
        "EB": "Eb",
        "D#": "D#",
        "DSHARP": "D#",
        "E": "E",
        "F": "F",
        "GB": "Gb",
        "F#": "F#",
        "FSHARP": "F#",
        "G": "G",
        "AB": "Ab",
        "G#": "G#",
        "GSHARP": "G#",
        "A": "A",
        "BB": "Bb",
        "A#": "A#",
        "ASHARP": "A#",
        "B": "B",
    }
    if upper in aliases:
        return aliases[upper]
    raise ValueError(
        f"Invalid key {key!r}. Use note names like Bb, Eb, F#, Gb, or F-sharp."
    )


@dataclass
class InterpretedTags:
    parts: list[str]
    part: str
    score_key: str | None = None
    score_mode: str | None = None
    variations: list[str] = field(default_factory=list)
    free_tags: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    noise: list[str] = field(default_factory=list)
    # False for big-band jumbo and combined-set scores: listed, but no
    # substitution chain ever reaches them.
    selectable: bool = True


# Big-band conductor scores: every part in sequence in one file, named for
# no single instrument.
_CONDUCTOR_NAMES = {"conductor", "score", "condensed score"}

# Bare Bb/Eb name a group in combo. In big band the group rule is off for
# them and they are the score key (Sweet Home Chicago.Bb).
_BARE_SCORE_KEYS = {"bb": "Bb", "eb": "Eb"}

# Priority per library type. 0 disables the rule. Within one span, higher
# is tried first; equal priority follows this order. The longest span still
# wins over a higher-priority shorter span. Rows without a span matcher
# (jumbo default, and the rest of the big-band profile) are declared here
# and consulted only where a matcher exists.
RULES: dict[str, dict[str, int]] = {
    "explicit score key": {"combo": 1, "big band": 1},
    "instrument": {"combo": 1, "big band": 1},
    "group": {"combo": 2, "big band": 1},
    "bare score key": {"combo": 0, "big band": 1},
    "conductor": {"combo": 0, "big band": 1},
    "kind": {"combo": 1, "big band": 1},
    "page split": {"combo": 1, "big band": 1},
    "noise": {"combo": 1, "big band": 1},
    "version": {"combo": 1, "big band": 1},
    "moved article": {"combo": 1, "big band": 1},
    "jumbo default": {"combo": 0, "big band": 1},
    "title initials": {"combo": 0, "big band": 1},
    "trailing key": {"combo": 0, "big band": 1},
    "big-band instruments": {"combo": 0, "big band": 1},
    "arranger": {"combo": 0, "big band": 1},
    "arrangement version": {"combo": 0, "big band": 1},
}

_RULE_INDEX = {name: index for index, name in enumerate(RULES)}

# Tag-span matchers. The other rows are profile declarations: jumbo default
# is applied when choosing the part, and the rest have no matcher yet.
_SPAN_MATCHERS = frozenset({
    "explicit score key",
    "instrument",
    "group",
    "bare score key",
    "conductor",
    "kind",
    "page split",
    "noise",
    "version",
})


def rule_priority(rule: str, library_type: str) -> int:
    """Priority of a naming rule for a library type; 0 disables it."""
    return RULES[rule][library_type]


def _span_rules(library_type: str) -> list[str]:
    """Enabled span matchers, highest priority first, table order on ties."""
    ranked = sorted(
        (name for name in RULES if name in _SPAN_MATCHERS),
        key=lambda name: (-RULES[name][library_type], _RULE_INDEX[name]),
    )
    return [name for name in ranked if RULES[name][library_type] > 0]


def _tokenize_tag_text(tag_text: str) -> list[str]:
    # Split on hard boundaries: ' - ', '-', '.', ',', '(', ')', '[', ']', ':', '/', '+', '&', '|'
    # Keep words; hard separators become token boundaries (spaces already normalized).
    # normalize first, then split on spaces.
    norm = normalize_text(tag_text)
    return [t for t in norm.split(" ") if t]


def interpret_remainder(
    tag_text: str, allow_dual: bool = True, library_type: str = "combo"
) -> InterpretedTags:
    """Interpret tag text right to left. The longest span wins, then priority.

    `RULES` gives each rule's priority for `library_type`; 0 skips the rule.
    A bare Bb or Eb is a group part in combo and the score key in big band.
    In a big-band library a conductor score is a part of its own, and an
    untagged remainder defaults to the full score instead of Concert.
    """
    tokens = _tokenize_tag_text(tag_text)
    n = len(tokens)
    consumed = [False] * n
    rules_fired: list[str] = []
    score_key: str | None = None
    score_mode: str | None = None
    instruments_found: list[tuple[int, str, str | None]] = []  # index, canon, role?
    conductors_found: list[int] = []
    groups_found: list[tuple[int, str]] = []
    kinds_found: list[tuple[int, str]] = []
    versions: list[str] = []
    noise_positions: list[int] = []
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

    span_rules = _span_rules(library_type)
    bare_key_on = rule_priority("bare score key", library_type) > 0

    def take(start: int, end: int) -> None:
        for k in range(start, end + 1):
            consumed[k] = True

    def apply(rule: str, span: int, start: int) -> bool:
        """Match one rule on the span ending at i and record the hit."""
        nonlocal score_key, score_mode
        phrase = " ".join(tokens[start : i + 1])
        tok = tokens[i]
        if rule == "explicit score key":
            if span == 2 and tokens[start] == "concert":
                key = _parse_key_token(tokens[i])
                if key:
                    score_key = key
                    rules_fired.append(f"explicit score key: {phrase}")
                    take(start, i)
                    return True
            if span == 2 and tokens[start] == "in":
                key = _parse_key_token(tokens[i])
                if key:
                    score_key = key
                    rules_fired.append(f"explicit score key: {phrase}")
                    take(start, i)
                    return True
            if span == 2 and tokens[i] in ("major", "minor"):
                key = _parse_key_token(tokens[start])
                if key:
                    score_key = key
                    score_mode = tokens[i]
                    rules_fired.append(f"explicit score key: {phrase}")
                    take(start, i)
                    return True
            # <key>m (gm, cm). "drm" is not a key base, so drums stay drums.
            if span == 1 and len(tok) >= 2 and tok.endswith("m"):
                base = tok[:-1]
                key = _parse_key_token(base)
                if key and base in _VALID_KEYS or base in ("g", "c", "d", "a", "e", "b"):
                    score_key = key
                    score_mode = "minor"
                    rules_fired.append(f"explicit score key: {phrase}")
                    consumed[i] = True
                    return True
            return False
        if rule == "instrument":
            if re.fullmatch(r"\d{1,2}", tok):
                inst_phrase = " ".join(tokens[start:i])
                canon_group = _ALIAS_MAP.get(inst_phrase)
                if canon_group:
                    canon, _ = canon_group
                    instruments_found.append((start, canon, tok))
                    rules_fired.append(f"instrument role: {phrase}")
                    take(start, i)
                    return True
            if span >= 2 and phrase in _ALIAS_MAP:
                canon, _ = _ALIAS_MAP[phrase]
                instruments_found.append((start, canon, None))
                rules_fired.append(f"instrument: {phrase}")
                take(start, i)
                return True
            if span == 1 and tok in _ALIAS_MAP:
                canon, _ = _ALIAS_MAP[tok]
                instruments_found.append((i, canon, None))
                rules_fired.append(f"instrument: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "group":
            # Bare Bb/Eb belong to the score-key rule when that rule is on.
            if phrase in _BARE_SCORE_KEYS and bare_key_on:
                return False
            if span >= 2 and phrase in _GROUPS:
                groups_found.append((start, _GROUPS[phrase]))
                rules_fired.append(f"group: {phrase}")
                take(start, i)
                return True
            if span == 1 and tok in _GROUPS:
                groups_found.append((i, _GROUPS[tok]))
                rules_fired.append(f"group: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "bare score key":
            if span == 1 and tok in _BARE_SCORE_KEYS:
                score_key = _BARE_SCORE_KEYS[tok]
                rules_fired.append(f"score key: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "conductor":
            if span >= 2 and phrase in _CONDUCTOR_NAMES:
                conductors_found.append(start)
                rules_fired.append(f"conductor: {phrase}")
                take(start, i)
                return True
            if span == 1 and tok in _CONDUCTOR_NAMES:
                conductors_found.append(i)
                rules_fired.append(f"conductor: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "kind":
            if span >= 2 and phrase in _KINDS:
                kinds_found.append((start, _KINDS[phrase]))
                rules_fired.append(f"kind: {phrase}")
                take(start, i)
                return True
            if span == 1 and tok in _KINDS:
                kinds_found.append((i, _KINDS[tok]))
                rules_fired.append(f"kind: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "page split":
            if (
                span == 2
                and tokens[start] in ("pg", "p", "page")
                and re.fullmatch(r"\d{1,3}", tok)
            ):
                rules_fired.append(f"page split: {phrase}")
                take(start, i)
                return True
            if span == 1 and tok in ("pg", "p", "page"):
                rules_fired.append(f"page split: {tok}")
                consumed[i] = True
                return True
            if span == 1 and re.fullmatch(r"p\d{1,3}", tok):
                rules_fired.append(f"page split: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "version":
            if span == 1 and re.fullmatch(r"v\d+", tok):
                versions.append(tok)
                rules_fired.append(f"version: {tok}")
                consumed[i] = True
                return True
            return False
        if rule == "noise":
            if span == 1 and tok in _FILLER_WORDS:
                rules_fired.append(f"filler: {tok}")
                consumed[i] = True
                return True
            if span == 1 and (re.fullmatch(r"\d{1,3}", tok) or tok in _NOISE_WORDS):
                rules_fired.append(f"noise: {tok}")
                noise_positions.append(i)
                consumed[i] = True
                return True
            return False
        return False

    i = n - 1
    while i >= 0:
        if consumed[i]:
            i -= 1
            continue
        # Longest span first; within a span, rule priority, then table order.
        matched = False
        for span in (3, 2, 1):
            start = i - span + 1
            if start < 0 or any(consumed[start : i + 1]):
                continue
            for rule in span_rules:
                if apply(rule, span, start):
                    matched = True
                    break
            if matched:
                break
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
    noise = [tokens[k] for k in sorted(noise_positions) if k - 1 not in roles]

    # Part determination.
    defaulted = False
    if dual:
        parts = dual
        part = "/".join(dual)
    elif instruments_found:
        # Rightmost instrument (largest index).
        instruments_found.sort(key=lambda x: x[0])
        pos, canon, role = instruments_found[-1]
        part = f"{canon} {role}" if role else canon
        parts = [part]
        variations_out = [c.lower() for _, c, _ in instruments_found[:-1]]
        # A role in parentheses right after an instrument is the role that
        # instrument plays: "tenor (horn 2)" is Tenor Sax as Horn 2.
        paren_role = re.search(r"\(\s*horn\s*(\d{1,2})\s*\)", tag_text, re.IGNORECASE)
        if (
            paren_role
            and canon == "Horn"
            and role == paren_role.group(1)
            and len(instruments_found) >= 2
        ):
            _, player, player_role = instruments_found[-2]
            part = f"{player} {player_role}" if player_role else player
            parts = [part, f"Horn {role}"]
            variations_out = [c.lower() for _, c, _ in instruments_found[:-2]]
    elif conductors_found:
        part = "Conductor"
        parts = [part]
        variations_out = []
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
        defaulted = True
        jumbo = rule_priority("jumbo default", library_type) > 0
        part = BIG_BAND_DEFAULT_PART if jumbo else COMBO_DEFAULT_PART
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
        variations=(variations_out if instruments_found else []) + versions,
        free_tags=free_tags,
        rules_fired=rules_fired,
        noise=noise,
        selectable=not (
            rule_priority("jumbo default", library_type) > 0
            and (defaulted or part in _GROUPS.values())
        ),
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
    return normalize_text(hit[0]) if hit else None


def match_prefix_boundary(layout_title: str, name_text: str) -> tuple[str, bool, bool] | None:
    """Return (remainder, is_hard_boundary, remainder_has_parens) or None.

    The remainder keeps its punctuation (lowercased), so interpretation can
    see parentheses. Hard separators: ' - ', '-', '.', ',', '(', ')', '[', ']', ':'. Soft is a
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
    return (rest_original.strip(), is_hard, has_parens)


def longer_title_name(title: str, name: str) -> bool:
    """True when `name` starts with `title` and continues with more words.

    The extra words are the rest of a shortened title, not tags
    (`I'm Gonna Sit Right Down` → `…and Write Myself a Letter`).
    """
    hit = match_prefix_boundary(title, name)
    if hit is None:
        return False
    return bool(normalize_text(hit[0]).split())


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


def _folder_score_key(folder: str | None, library_type: str = "combo") -> str | None:
    """Score key named by a title folder (`- Concert F`, `(in F)`, `- Bb`)."""
    if not folder:
        return None
    return interpret_remainder(folder, library_type=library_type).score_key


def is_known_part(requested: str) -> bool:
    """Whether a --part value names an instrument (with role), group or kind."""
    part = _base_instrument(canonical_part_name(requested))
    return (
        part in _INSTRUMENT_GROUPS
        or part in _GROUPS.values()
        or part in _KINDS.values()
        or part in KEY_GROUPS
        or part == COMBO_DEFAULT_PART
    )


def parse_part_name(raw: str, title: str | None = None) -> str:
    """Validate a part name from `--part`, a named-part definition or a chart
    mapping (keys and values), and return its canonical form.

    Unknown names are an error naming the value and, for a chart, its title.
    """
    if is_known_part(raw):
        return canonical_part_name(raw)
    where = f' for "{title}"' if title is not None else ""
    raise ValueError(
        f'Unknown part "{raw.strip()}"{where}: not an instrument, group or kind, '
        "and not named in the layout config"
    )


@dataclass
class TitleCandidate:
    remainder: str
    parts: list[str]
    part: str
    score_key: str | None = None
    variations: list[str] = field(default_factory=list)
    free_tags: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    noise: list[str] = field(default_factory=list)
    # False for big-band jumbo and combined-set scores: listed, but no
    # substitution chain ever reaches them.
    selectable: bool = True
    # Set for part-index entries: pages of the part inside its file.
    page: int | None = None
    length: int | None = None
    # All ranges, in list order.
    ranges: list[tuple[int, int]] | None = None

    @property
    def tag_count(self) -> int:
        """Extra words: score key, variations, free tags and noise words."""
        return (
            (1 if self.score_key else 0)
            + len(self.variations)
            + len(self.free_tags)
            + len(self.noise)
        )


def interpret_filename(stem: str) -> InterpretedTags:
    """A filename read whole as tags, however its name reads (explicit index
    `file:`): variations, free tags and score key come from the name, while
    the parts come from the index entry."""
    return interpret_remainder(clean_stem(stem))


def match_candidate(
    title: str,
    stem: str,
    folder: str | None = None,
    library_type: str = "combo",
) -> TitleCandidate | None:
    """Match layout title against stem or folder, interpreting remainder tags.

    Hard-boundary remainders accept free tags as variations; soft-boundary
    remainders require all-tag content (Morning Piano yes, Morning Dance no).
    """
    copy_noise = ["copy of"] if _COPY_PREFIX.match(stem.strip()) else []
    stem_hit = match_prefix_boundary(title, stem)
    if stem_hit is not None:
        stem_remainder, is_hard, has_parens = stem_hit
        interp = interpret_remainder(
            stem_remainder, allow_dual=not has_parens, library_type=library_type
        )
        if not (interp.free_tags and not is_hard):
            score_key = interp.score_key or _folder_score_key(folder, library_type)
            return TitleCandidate(
                remainder=stem_remainder,
                parts=interp.parts,
                part=interp.part,
                score_key=score_key,
                variations=interp.variations,
                free_tags=interp.free_tags,
                rules_fired=interp.rules_fired,
                noise=copy_noise + interp.noise,
                selectable=interp.selectable,
            )
        # Soft-boundary remainder with free tags: not this title via the stem
        # (e.g. "Mack_The_Knife__Bobby_Darin Score-Piano" slug words). Fall
        # through to folder matching below.
    if folder:
        folder_hit = match_prefix_boundary(title, folder)
        if folder_hit is not None:
            folder_remainder, folder_hard, folder_parens = folder_hit
            folder_interp = interpret_remainder(
                folder_remainder,
                allow_dual=not folder_parens,
                library_type=library_type,
            )
            if folder_interp.free_tags and not folder_hard:
                return None
            return stem_candidate(
                stem, folder, folder_interp.score_key, library_type
            )
    return None


def _title_initials(name: str) -> str:
    """First letter of each word. `Fly Me To The Moon` is `fmttm`."""
    return "".join(word[0] for word in normalize_text(name).split() if word)


def _initials_tags(stem: str, folder: str | None, library_type: str) -> str | None:
    """Tags after a title folder's initials, or None when that rule is off.

    The initials need at least three letters. Digits are split from the
    words they were glued to (`bass1` → `bass 1`).
    """
    if folder is None or rule_priority("title initials", library_type) <= 0:
        return None
    initials = _title_initials(folder)
    if len(initials) < 3:
        return None
    squashed = squash_text(clean_stem(stem))
    if not squashed.startswith(initials):
        return None
    rest = squashed[len(initials) :]
    return re.sub(r"(?<=\d)(?=\D)|(?<=\D)(?=\d)", " ", rest)


def _piano_page(interp: InterpretedTags) -> InterpretedTags:
    """A number glued to Piano is a page of that part, not Piano N."""
    match = re.fullmatch(r"Piano (\d+)", interp.part)
    if match is None:
        return interp
    return replace(
        interp,
        parts=["Piano"],
        part="Piano",
        rules_fired=[*interp.rules_fired, f"page split: {match.group(1)}"],
    )


def stem_candidate(
    stem: str,
    folder: str | None = None,
    folder_key: str | None = None,
    library_type: str = "combo",
) -> TitleCandidate:
    """A file whose title is known from its folder: the whole stem is read
    as tags, and words no rule takes are ignored.

    In a big-band library a stem that starts with the folder title's
    initials is that title; the rest of the name is the tags.
    """
    initials = _initials_tags(stem, folder, library_type)
    text = clean_stem(stem) if initials is None else initials
    interp = interpret_remainder(text, library_type=library_type)
    if initials is not None:
        interp = _piano_page(interp)
        interp = replace(interp, rules_fired=["title initials", *interp.rules_fired])
    return TitleCandidate(
        remainder=clean_stem(stem),
        parts=interp.parts,
        part=interp.part,
        score_key=interp.score_key or folder_key or _folder_score_key(folder, library_type),
        variations=interp.variations,
        free_tags=[],
        rules_fired=interp.rules_fired,
        noise=(["copy of"] if _COPY_PREFIX.match(stem.strip()) else []) + interp.noise,
        selectable=interp.selectable,
    )


def chain_for_part(requested: str) -> list[str]:
    """Substitution chain for a requested part, earliest first."""
    part = canonical_part_name(requested)
    if part == "Piano":
        return _chain("Piano", "Rhythm", *_CONCERT_RANK, *_FALLBACK_REST)
    if part == "Guitar":
        return _chain("Guitar", "Piano", "Rhythm", *_CONCERT_RANK, *_FALLBACK_REST)
    if part == "Bass":
        return _chain("Bass", "Rhythm", *_CONCERT_RANK, *_FALLBACK_REST)
    if part == "Drums":
        return _chain("Drums", "Rhythm", *_CONCERT_RANK, *_FALLBACK_REST)
    if part == "Rhythm":
        return _chain("Rhythm", *_CONCERT_FALLBACK)
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
        for fallback in [*_CONCERT_RANK, *_FALLBACK_REST]:
            if fallback not in steps:
                steps.append(fallback)
        return steps
    # Group or unknown: direct then Concert fallback.
    steps = [part]
    for fallback in [*_CONCERT_RANK, *_FALLBACK_REST]:
        if fallback not in steps:
            steps.append(fallback)
    return steps


KEY_GROUPS = {"Bb", "Eb", "Bass Clef", "F"}


def rank_chain(requested: str) -> list[list[str]]:
    """Substitution chain grouped into ranks, best first.

    A key group (Bb, Eb, Bass Clef, F) shares its instrument's rank: a Bb
    chart is as good as a Tenor Sax chart. Concert, Lead Sheet and Rhythm
    form one rank, as do Piano and Guitar. Every other step is its own rank.
    """
    ranks: list[list[str]] = []
    for step in chain_for_part(requested):
        prev = ranks[-1][-1] if ranks and ranks[-1] else None
        if ranks and (
            step in KEY_GROUPS
            or (step == "Lead Sheet" and prev == "Concert")
            or (step == "Rhythm" and prev == "Lead Sheet")
            or (step == "Guitar" and prev == "Piano")
        ):
            ranks[-1].append(step)
        else:
            ranks.append([step])
    return ranks
