from __future__ import annotations

from geoinsight.processing.cleaner import _is_duplicate, normalise_place_name


def test_accented_name_normalisation_for_deduplication():
    assert normalise_place_name(" Latt\u00e9s! ") == "lattes"
    seen = [(normalise_place_name("Latt\u00e9s"), 50.9350, -1.3960)]

    assert _is_duplicate(normalise_place_name("Lattes"), 50.9351, -1.3960, seen)
