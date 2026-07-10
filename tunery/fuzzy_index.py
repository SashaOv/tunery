import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz, process


def normalize_key(text: str) -> str:
    """Normalize text for fuzzy matching."""
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return " ".join(normalized.split())


def key_variants(key: str) -> Iterable[str]:
    """Yield searchable variants for a key."""
    yield key


def strip_parenthetical_suffix(text: str) -> str:
    """Strip a parenthetical suffix from text."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()


def title_variants(title: str) -> Iterable[str]:
    """Yield searchable variants for a title."""
    yield title

    stripped = strip_parenthetical_suffix(title)
    if stripped != title:
        yield stripped


@dataclass(frozen=True)
class MatchResult[T]:
    score: float
    match: str
    item: T


class FuzzyIndex[T]:
    def __init__(
        self,
        *,
        normalizer: Callable[[str], str] = normalize_key,
        variants: Callable[[str], Iterable[str]] = key_variants,
        scorer: Any = fuzz.WRatio,
    ) -> None:
        self._normalizer = normalizer
        self._variants = variants
        self._scorer = scorer
        self._entries: dict[str, list[tuple[str, T]]] = {}

    def add(self, key: str, item: T) -> None:
        """Add an item to the index."""
        seen_variants: set[str] = set()
        for variant in self._variants(key):
            normalized = self._normalizer(variant)
            if not normalized or normalized in seen_variants:
                continue
            seen_variants.add(normalized)
            self._entries.setdefault(normalized, []).append((variant, item))

    def match(
        self,
        query: str,
        *,
        score_cutoff: int = 90,
        limit: int = 5,
    ) -> list[MatchResult[T]]:
        """Return a list of matches for the query."""
        if not self._entries:
            return []

        normalized_query = self._normalizer(query)
        matches = process.extract(
            normalized_query,
            list(self._entries),
            scorer=self._scorer,
            score_cutoff=score_cutoff,
            limit=limit,
        )

        results: list[MatchResult[T]] = []
        for normalized_match, score, _ in matches:
            for original_key, item in self._entries[normalized_match]:
                results.append(
                    MatchResult(score=score, match=original_key, item=item)
                )
                if len(results) >= limit:
                    return results

        return results
