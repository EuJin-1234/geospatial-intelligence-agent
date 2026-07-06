from __future__ import annotations

import logging
from pathlib import Path

from geoinsight.config import LegacyAppConfig, ensure_legacy_data_dirs, load_legacy_config

LOGGER = logging.getLogger(__name__)
OSM_TAGS = {"amenity": ["cafe", "library", "restaurant", "fast_food", "pub", "bar", "bus_station", "parking", "bicycle_parking", "university", "school"], "leisure": ["park", "garden", "sports_centre"], "shop": True, "public_transport": True, "tourism": True}


def load_osm_features(lat: float, lon: float, radius_m: int, force_refresh: bool = False, config: LegacyAppConfig | None = None):
    config = config or load_legacy_config()
    ensure_legacy_data_dirs(config)
    if config.raw_osm_path.exists() and not force_refresh:
        return _read_geojson(config.raw_osm_path)
    try:
        import osmnx as ox
        gdf = ox.features_from_point((lat, lon), tags=OSM_TAGS, dist=radius_m)
    except Exception as exc:
        LOGGER.warning("OSM download failed: %s", exc)
        return _empty_geodataframe()
    if gdf is None or gdf.empty:
        return _empty_geodataframe()
    gdf = gdf.reset_index()
    try:
        gdf.to_file(config.raw_osm_path, driver="GeoJSON")
    except Exception as exc:
        LOGGER.warning("Could not save raw OSM features: %s", exc)
    return gdf


def _read_geojson(path: Path):
    import geopandas as gpd
    return gpd.read_file(path)


def _empty_geodataframe():
    try:
        import geopandas as gpd
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    except ImportError:
        return []