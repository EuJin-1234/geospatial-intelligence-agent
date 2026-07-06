from __future__ import annotations

import json
from pathlib import Path

from geoinsight.config import load_config as load_legacy_config
from geoinsight.schemas import PlaceRecord

from geoinsight.config import ensure_data_dirs, load_config
from geoinsight.eda.spatial_distribution import spatial_overview
from geoinsight.eda.summary_stats import amenity_theme_overview, dataset_overview
from geoinsight.schemas import EDAReport


def build_eda_report(
    records: list[PlaceRecord],
    raw_feature_count: int | None = None,
    invalid_geometry_count: int = 0,
    write_charts: bool = True,
) -> EDAReport:
    config = load_config()
    ensure_data_dirs(config)
    spatial = spatial_overview(records)
    area_sq_km = None
    if spatial.get("bounding_box"):
        area_sq_km = spatial["bounding_box"].get("area_sq_km_estimate")
    report = EDAReport(
        dataset_overview=dataset_overview(records, raw_feature_count, invalid_geometry_count),
        spatial_overview=spatial,
        amenity_theme_overview=amenity_theme_overview(records, area_sq_km),
    )
    save_eda_report(report, config.eda_summary_json_path, config.eda_summary_md_path)
    if write_charts:
        write_eda_charts(report, records, config.reports_dir)
    return report


def load_records(path: Path | None = None) -> list[PlaceRecord]:
    path = path or load_legacy_config().processed_records_path
    if not path.exists():
        return []
    return [PlaceRecord.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_eda_report(report: EDAReport, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")


def write_eda_charts(report: EDAReport, records: list[PlaceRecord], reports_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    reports_dir.mkdir(parents=True, exist_ok=True)
    _bar_chart(
        plt,
        report.dataset_overview.get("category_counts", {}),
        "Category Counts",
        reports_dir / "category_counts.png",
    )
    _bar_chart(
        plt,
        report.amenity_theme_overview.get("top_themes", {}),
        "Theme Counts",
        reports_dir / "theme_counts.png",
    )
    if records:
        plt.figure(figsize=(7, 4))
        plt.hist([record.distance_to_origin_m for record in records], bins=12)
        plt.title("Distance Distribution")
        plt.xlabel("Distance from origin (m)")
        plt.ylabel("Places")
        plt.tight_layout()
        plt.savefig(reports_dir / "distance_distribution.png")
        plt.close()


def _bar_chart(plt, values: dict, title: str, path: Path) -> None:
    if not values:
        return
    items = list(values.items())[:12]
    labels = [item[0] for item in items]
    counts = [item[1] for item in items]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _markdown(report: EDAReport) -> str:
    overview = report.dataset_overview
    spatial = report.spatial_overview
    themes = report.amenity_theme_overview
    return "\n".join(
        [
            "# GeoInsight EDA Summary",
            "",
            f"- Raw features: {overview.get('raw_features', 0)}",
            f"- Cleaned places: {overview.get('cleaned_places', 0)}",
            f"- Named places: {overview.get('named_places', 0)}",
            f"- Unnamed places: {overview.get('unnamed_places', 0)}",
            f"- Invalid/removed geometries: {overview.get('invalid_removed_geometries', 0)}",
            "",
            "## Top Categories",
            json.dumps(overview.get("category_counts", {}), indent=2),
            "",
            "## Spatial Overview",
            json.dumps(spatial, indent=2),
            "",
            "## Amenity And Theme Overview",
            json.dumps(themes, indent=2),
            "",
        ]
    )
