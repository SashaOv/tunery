from tunery.parts import (
    clean_stem,
    interpret_remainder,
    match_prefix,
    normalize_text,
    parse_filename,
    squash_text,
)


def test_normalization_squashed_equal() -> None:
    assert squash_text("SoWhat") == squash_text("So What")
    assert squash_text("Mo’ Better Blues") == squash_text("Mo Better Blues")
    assert normalize_text("Bass_Clarinet") == "bass clarinet"


def test_clean_stem() -> None:
    assert clean_stem("Finale 2004a - [My Tune]") == "My Tune"
    assert clean_stem("Copy of My Tune") == "My Tune"
    assert clean_stem("Copy (2) of My Tune") == "My Tune"
    assert clean_stem("My Tune (1)") == "My Tune"
    assert clean_stem("Mo Better Blues-Piano_(1)") == "Mo Better Blues-Piano"


def test_morning_soft_boundary() -> None:
    piano = interpret_remainder("Piano")
    assert piano.part == "Piano"
    assert piano.free_tags == []

    dance = interpret_remainder("Dance")
    assert dance.free_tags == ["dance"]
    assert dance.part == "Concert"


def test_title_keeps_trailing_word() -> None:
    assert match_prefix("Partido Alto", "Partido Alto") == ""
    assert match_prefix("Fascinating Rhythm", "Fascinating Rhythm") == ""
    assert match_prefix("Morning", "Morning Piano") == "piano"


def test_multiword_spans() -> None:
    ceora = interpret_remainder("Ceora Bb Tenor")
    assert ceora.part == "Tenor Sax"

    mack = interpret_remainder("Mack Bb Solo")
    assert mack.part == "Bb"


def test_earlier_instrument_is_variation() -> None:
    item = parse_filename("All of Me (Vocal)-Piano")
    assert item.part == "Piano"
    assert "vocal" in [v.lower() for v in item.variations]


def test_dual_scores() -> None:
    for stem in ("Piano:Guitar", "piano-guitar", "PianoGuitar", "pnoguit"):
        item = parse_filename(stem)
        assert set(item.parts) == {"Piano", "Guitar"}, stem


def test_roles() -> None:
    assert interpret_remainder("Horn 1").part == "Horn 1"
    assert interpret_remainder("Horn 2").part == "Horn 2"
    assert interpret_remainder("Alto Sax 2").part == "Alto Sax 2"
    assert interpret_remainder("Norwegian Wood - Horn 1").part == "Horn 1"


def test_group_and_score_key() -> None:
    assert interpret_remainder("Bb").part == "Bb"
    assert interpret_remainder("Concert F").score_key == "F"
    item = parse_filename("slug-Piano", folder="Aint Misbehavin - Concert F")
    assert item.score_key == "F"
    assert item.part == "Piano"
    assert parse_filename("song-Concert Bb").score_key == "Bb"


def test_rhythm_aliases() -> None:
    assert interpret_remainder("Electric_Guitar").part == "Guitar"
    assert interpret_remainder("Electric Piano").part == "Piano"
    assert interpret_remainder("Pno").part == "Piano"
    assert interpret_remainder("Gtr").part == "Guitar"
    assert interpret_remainder("Bss").part == "Bass"
    assert interpret_remainder("Drm").part == "Drums"


def test_page_split_and_noise() -> None:
    assert interpret_remainder("P1").rules_fired[0].startswith("page split")
    assert interpret_remainder("Pg 2").rules_fired[0].startswith("page split")
    noisy = interpret_remainder("Tune v2 original rescan")
    assert noisy.free_tags == ["tune"]
    assert interpret_remainder("Song, The").free_tags == ["song"]


def test_kinds_and_melody() -> None:
    assert interpret_remainder("Lead Sheet").part == "Lead Sheet"
    assert interpret_remainder("Lyrics").part == "Lyrics"
    assert interpret_remainder("Melody").part == "Melody"


def test_vault_names() -> None:
    assert parse_filename("mack-Score-Piano").part == "Piano"
    assert parse_filename("tune-Guitar").part == "Guitar"
    assert parse_filename("tune-Bass").part == "Bass"
    assert parse_filename("soft-impressions-Bb").part == "Bb"
