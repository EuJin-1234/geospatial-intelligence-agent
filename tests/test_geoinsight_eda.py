from __future__ import annotations

from geoinsight.eda.report_builder import build_eda_report


def test_eda_summary_generation(make_record, monkeypatch, tmp_path):
    from geoinsight import config as config_module

    config = config_module.load_config()
    test_config = config.__class__(
        **{
            **config.__dict__,
            "reports_dir": tmp_path,
            "eda_summary_json_path": tmp_path / "eda_summary.json",
            "eda_summary_md_path": tmp_path / "eda_summary.md",
        }
    )
    monkeypatch.setattr("geoinsight.eda.report_builder.load_config", lambda: test_config)
    monkeypatch.setattr("geoinsight.eda.report_builder.ensure_data_dirs", lambda config=None: None)

    report = build_eda_report([make_record(name="Cafe", category="cafe")], write_charts=False)

    assert report.dataset_overview["cleaned_places"] == 1
    assert (tmp_path / "eda_summary.json").exists()
