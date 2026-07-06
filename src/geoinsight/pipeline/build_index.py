from __future__ import annotations

import logging
from pathlib import Path

from geoinsight.config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_M, load_config
from geoinsight.ingestion.osm_loader import load_osm_features
from geoinsight.processing.cleaner import clean_osm_features
from geoinsight.processing.description_builder import add_descriptions, save_records
from geoinsight.processing.spatial_features import add_spatial_features
from geoinsight.processing.spatial_relations import build_spatial_relations
from geoinsight.indexing.vector_store import FaissVectorStore

from geoinsight.indexing.embedder import create_embedder

LOGGER = logging.getLogger(__name__)


def build_index(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    radius_m: int = DEFAULT_RADIUS_M,
    force_refresh: bool = False,
    embedding_model: str | None = None,
    embedding_backend: str | None = None,
    onnx_model_dir: str | Path | None = None,
) -> int:
    """Build the GeoInsight vector index with optional edge embedding backend."""
    config = load_config()
    legacy_config = config.legacy
    gdf = load_osm_features(lat, lon, radius_m, force_refresh=force_refresh, config=legacy_config)
    records = clean_osm_features(gdf)
    records = add_spatial_features(records, lat, lon)
    records = build_spatial_relations(records)
    records = add_descriptions(records)
    save_records(records, config=legacy_config)

    if not records:
        LOGGER.warning("No records were produced; skipping vector index build")
        return 0

    embedder = create_embedder(
        embedding_model or legacy_config.embedding_model_name,
        device=legacy_config.embedding_device,
        backend=embedding_backend or legacy_config.embedding_backend,
        onnx_model_dir=onnx_model_dir or legacy_config.onnx_embedding_model_dir,
    )
    store = FaissVectorStore(embedder=embedder, config=legacy_config)
    store.build_index(records)
    store.save()
    LOGGER.info("Built vector index for %s records", len(records))
    return len(records)


__all__ = ["build_index"]