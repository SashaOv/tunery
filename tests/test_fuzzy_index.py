from pathlib import Path

from tunery.fuzzy_index import (
    FuzzyIndex,
    normalize_key,
    strip_parenthetical_suffix,
    title_variants,
)


def test_normalize_key_removes_punctuation_and_normalizes_whitespace() -> None:
    assert normalize_key("  L-O-V-E  ") == "love"
    assert normalize_key("Feel Like Makin' Love") == "feel like makin love"
    assert normalize_key("Blue   In\tGreen") == "blue in green"


def test_strip_parenthetical_suffix_removes_trailing_parenthetical() -> None:
    assert strip_parenthetical_suffix("Lover Man (Oh, Where Can You Be?)") == "Lover Man"
    assert strip_parenthetical_suffix("Blue In Green") == "Blue In Green"


def test_fuzzy_index_returns_payload_with_match() -> None:
    index = FuzzyIndex[Path]()
    source = Path("The Girl from Ipanema.pdf")

    index.add("The Girl from Ipanema", source)

    results = index.match("Girl from Ipanema", score_cutoff=80)

    assert results
    assert results[0].match == "The Girl from Ipanema"
    assert results[0].item == source
    assert results[0].score >= 90


def test_fuzzy_index_respects_score_cutoff() -> None:
    index = FuzzyIndex[str]()
    index.add("Blue In Green", "blue")

    assert index.match("Completely Different", score_cutoff=90) == []


def test_fuzzy_index_preserves_items_for_duplicate_normalized_keys() -> None:
    index = FuzzyIndex[str]()
    index.add("L-O-V-E", "punctuated")
    index.add("Love", "plain")

    results = index.match("love", score_cutoff=100, limit=10)

    assert [(result.match, result.item) for result in results] == [
        ("L-O-V-E", "punctuated"),
        ("Love", "plain"),
    ]


def test_fuzzy_index_can_match_variant_key_to_canonical_item() -> None:
    index = FuzzyIndex[str](variants=title_variants)
    index.add("Lover Man (Oh, Where Can You Be?)", "full title")

    results = index.match("Lover Man", score_cutoff=90)

    assert results[0].match == "Lover Man"
    assert results[0].item == "full title"
