from tunery.parts import (
    chain_for_part,
    clean_stem,
    fuzzy_title_prefix,
    interpret_remainder,
    match_candidate,
    match_prefix,
    normalize_text,
    parse_filename,
    squash_text,
    stem_candidate,
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


def test_title_initials_read_the_rest_as_tags() -> None:
    folder = "Fly Me To The Moon"
    bass = stem_candidate("fmttmbass", folder, library_type="big band")
    assert bass.part == "Bass"
    numbered = stem_candidate("fmttmbass1", folder, library_type="big band")
    assert numbered.part == "Bass 1"
    piano = stem_candidate("fmttmpno1", folder, library_type="big band")
    assert piano.part == "Piano"
    assert any(rule.startswith("page split") for rule in piano.rules_fired)
    # Two-letter initials stay off (Glenn's Den is `gd`).
    short = stem_candidate("gdbass", "Glenn's Den", library_type="big band")
    assert short.part == "Score"


def test_combo_title_initials_stay_off() -> None:
    ode = match_candidate("Ode", "Ode.bass", "Ode")
    assert ode is not None
    assert ode.part == "Bass"
    assert "title initials" not in ode.rules_fired
    # Three-letter initials that prefix a name still do not apply in combo.
    glued = stem_candidate("fmttmbass", "Fly Me To The Moon")
    assert glued.part == "Concert"


def test_bare_bb_eb_is_score_key_in_big_band_only() -> None:
    """A bare Bb/Eb is the score key in big band, and a group part in combo."""
    for token in ("Bb", "Eb"):
        band = match_candidate(
            "Sweet Home Chicago",
            f"Sweet Home Chicago.{token}",
            library_type="big band",
        )
        assert band is not None
        assert band.part == "Score", token
        assert band.score_key == token
        assert band.selectable is False
        combo = match_candidate(
            "Sweet Home Chicago", f"Sweet Home Chicago.{token}"
        )
        assert combo is not None
        assert combo.part == token
        assert combo.score_key is None
        assert combo.selectable is True


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


def test_bass_clarinet_distinct_from_bass() -> None:
    assert interpret_remainder("Bass Clarinet").part == "Bass Clarinet"
    assert interpret_remainder("Bass Trombone").part == "Bass Trombone"
    assert "Bass Clarinet" not in chain_for_part("Bass")
    assert "Bass" not in chain_for_part("Bass Clarinet")
    assert "Bass Trombone" not in chain_for_part("Bass")
    assert "Bass Trombone" not in chain_for_part("Trombone")


def test_bass_clef_serves_trombones_not_bass() -> None:
    assert interpret_remainder("Bass Clef").part == "Bass Clef"
    assert "Bass Clef" in chain_for_part("Trombone")
    assert "Bass Clef" in chain_for_part("Bass Trombone")
    assert "Bass Clef" not in chain_for_part("Bass")


def test_paren_instrument_is_variation_not_dual() -> None:
    r = interpret_remainder("(Vocal)-Piano")
    assert r.part == "Piano"
    assert r.parts == ["Piano"]
    assert "vocal" in [v.lower() for v in r.variations]


def test_fuzzy_title_prefix_matches_word_prefix() -> None:
    stem = "si-tu-vois-ma-mere-sidney-bechet-lead-sheet-C_Major"
    assert fuzzy_title_prefix("Si Tu Vois Ma Mare", stem) == "si tu vois ma mere"
    # Two misspellings (vois/voir, mare/mere) fall below the 90 cutoff.
    assert fuzzy_title_prefix("Si Tu Vois Ma Mare", "Si Tu Voir Ma Mere") is None


def test_fuzzy_title_prefix_rejects_inner_substring() -> None:
    assert fuzzy_title_prefix("Morning", "Softly As In A Morning Sunrise") is None
    assert fuzzy_title_prefix("Sunny", "On The Sunny Side Of The Street") is None


def test_unspecified_sax_is_not_concert() -> None:
    assert interpret_remainder("Tenro Sax part in Concert Gm)").part == "Saxophone"
    assert interpret_remainder("(Alto Sax part in Concert Gm)").part == "Alto Sax"


def test_parenthesized_role_is_played_by_preceding_instrument() -> None:
    r = interpret_remainder("tenor (horn 2)")
    assert r.part == "Tenor Sax"
    assert r.parts == ["Tenor Sax", "Horn 2"]
    assert r.variations == []


def test_version_is_a_variation() -> None:
    r = interpret_remainder("Tenor Sax.v2")
    assert r.part == "Tenor Sax"
    assert r.variations == ["v2"]


def test_noise_words_add_extra_words_but_fillers_do_not() -> None:
    plain = match_candidate("Creepin", "Creepin")
    noisy = match_candidate("Creepin", "Creepin.iRealPro")
    assert noisy.tag_count == plain.tag_count + 1
    assert match_candidate("Tune", "Tune - Bb part").tag_count == 0


def test_copy_of_counts_as_noise() -> None:
    original = match_candidate("Shaker Song", "Shaker Song - Tenor")
    copy = match_candidate("Shaker Song", "Copy of Shaker Song - Tenor")
    assert copy.part == original.part == "Tenor Sax"
    assert copy.tag_count == original.tag_count + 1


def test_parenthesized_role_survives_title_match() -> None:
    c = match_candidate("The Cape Verdean Blues", "The Cape Verdean Blues - tenor (horn 2)")
    assert (c.part, c.parts) == ("Tenor Sax", ["Tenor Sax", "Horn 2"])
