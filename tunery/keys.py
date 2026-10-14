"""Key signature handling for per-key setbook editions.

Key name normalization is copied from the pdfscore script
(`pdfscore.py`: `KEY_NUMBERS`, `KEY_ALIASES`, `_canonicalize_key_token`,
`normalize_key_name`) so tunery accepts the same `--keys` values without a
cross-repo dependency.
"""

import re
from pathlib import Path

CONCERT_KEY = "C"
CONCERT_SUFFIX = "Concert"

KEY_NUMBERS: dict[str, int] = {
    "C": 0,
    "Db": 1,
    "C#": 1,
    "D": 2,
    "Eb": 3,
    "D#": 3,
    "E": 4,
    "F": 5,
    "Gb": 6,
    "F#": 6,
    "G": 7,
    "Ab": 8,
    "G#": 8,
    "A": 9,
    "Bb": 10,
    "A#": 10,
    "B": 11,
}

KEY_ALIASES: dict[str, tuple[int, str]] = {
    "C": (0, "C"),
    "DB": (1, "Db"),
    "C#": (1, "C#"),
    "CSHARP": (1, "C#"),
    "D": (2, "D"),
    "EB": (3, "Eb"),
    "D#": (3, "D#"),
    "DSHARP": (3, "D#"),
    "E": (4, "E"),
    "F": (5, "F"),
    "GB": (6, "Gb"),
    "F#": (6, "F#"),
    "FSHARP": (6, "F#"),
    "G": (7, "G"),
    "AB": (8, "Ab"),
    "G#": (8, "G#"),
    "GSHARP": (8, "G#"),
    "A": (9, "A"),
    "BB": (10, "Bb"),
    "A#": (10, "A#"),
    "ASHARP": (10, "A#"),
    "B": (11, "B"),
}


def _canonicalize_key_token(key: str) -> str:
    token = key.strip().replace("♯", "#").replace("♭", "b")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"(?i)(?<=[A-Ga-g])-?sharp", "#", token)
    token = re.sub(r"(?i)(?<=[A-Ga-g])-?flat", "b", token)
    return token


def normalize_key_name(key: str) -> str:
    """Canonicalize a user-supplied key name (`bb` -> `Bb`)."""
    token = _canonicalize_key_token(key)
    if token.upper() == CONCERT_SUFFIX.upper():
        return CONCERT_KEY
    alias = KEY_ALIASES.get(token.upper())
    if alias is not None:
        return alias[1]

    raise ValueError(
        f"Invalid key {key!r}. Use note names like Bb, Eb, F#, Gb, or F-sharp."
    )


def is_concert(key: str) -> bool:
    """Check whether a canonical key name selects the concert edition."""
    return key == CONCERT_KEY


def expand_keys(keys: list[str]) -> list[str]:
    """Expand CLI key values into canonical key names, deduplicated in order.

    Each value may hold comma-separated keys. An empty input selects the
    concert edition.
    """
    expanded: list[str] = []
    for raw in keys:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            normalized = normalize_key_name(part)
            if normalized not in expanded:
                expanded.append(normalized)
    if not expanded:
        return [CONCERT_KEY]
    return expanded


def edition_suffix(key: str) -> str | None:
    """Filename suffix for a key edition, or None for the concert edition."""
    if is_concert(key):
        return None
    return key


def keyed_filename(filename: str, key: str) -> str | None:
    """Insert the key suffix before the `.pdf` extension, if applicable."""
    if is_concert(key):
        suffix = CONCERT_SUFFIX
    else:
        suffix = key
    path = Path(filename)
    if path.suffix.lower() != ".pdf":
        return None
    return str(path.with_name(f"{path.stem}.{suffix}.pdf"))


def output_path_for_key(output: Path, key: str) -> Path:
    """Derive the per-key setbook output path from the base output path."""
    suffix = edition_suffix(key)
    if suffix is None:
        return output
    return output.with_name(f"{output.stem}.{suffix}{output.suffix}")
