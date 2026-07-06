from __future__ import annotations

from geoinsight.config import load_config as load_legacy_config
from geoinsight.ingestion.osm_loader import load_osm_features
from geoinsight.processing.cleaner import clean_osm_features
from geoinsight.processing.description_builder import add_descriptions, save_records
from geoinsight.processing.spatial_features import add_spatial_features
from geoinsight.processing.spatial_relations import build_spatial_relations

from geoinsight.data_quality.validation import validate_place_records, validate_raw_geometries
from geoinsight.eda.report_builder import build_eda_report
from geoinsight.schemas import BuildReport, DatasetBuildOptions


def build_dataset(options: DatasetBuildOptions) -> BuildReport:
    legacy_config = load_legacy_config()
    gdf = load_osm_features(
        options.lat,
        options.lon,
        options.radius_m,
        force_refresh=options.force_refresh,
        config=legacy_config,
    )
    geometry_profile = validate_raw_geometries(gdf)
    records = clean_osm_features(gdf)
    records = add_spatial_features(records, options.lat, options.lon)
    records = build_spatial_relations(records)
    records = add_descriptions(records)
    save_records(records, config=legacy_config)
    validate_place_records(records)
    build_eda_report(
        records,
        raw_feature_count=geometry_profile["raw_features"],
        invalid_geometry_count=geometry_profile["invalid_geometry"] + geometry_profile["missing_geometry"],
    )
    return BuildReport(
        records_built=len(records),
        outputs={"records": str(legacy_config.processed_records_path)},
    )
