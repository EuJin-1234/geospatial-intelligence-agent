from __future__ import annotations

from geoinsight.data_quality.validation import validate_place_records


def test_data_quality_report_flags_warnings(make_record, tmp_path):
    records = [
        make_record(name="Unnamed place", category="place", themes=[], place_id="a"),
        make_record(name="Named Cafe", category="cafe", themes=["food"], place_id="b"),
    ]

    report = validate_place_records(records, tmp_path / "quality.json")

    assert report.total_records == 2
    assert report.warning_count > 0
    assert "missing_names" in report.warnings_by_type
