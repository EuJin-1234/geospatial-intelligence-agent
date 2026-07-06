from __future__ import annotations

from pathlib import Path

from geoinsight.eda.report_builder import write_eda_charts
from geoinsight.schemas import EDAReport


def write_summary_charts(report: EDAReport, records: list, reports_dir: Path) -> None:
    write_eda_charts(report, records, reports_dir)
