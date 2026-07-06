from __future__ import annotations

from pathlib import Path

from geoinsight.config import LegacyAppConfig, ensure_legacy_data_dirs, load_legacy_config
from geoinsight.processing.spatial_relations import relation_summary
from geoinsight.schemas import RetrievedPlace


class FoliumMapBuilder:
    def __init__(self, config: LegacyAppConfig | None = None):
        self.config = config or load_legacy_config()

    def build_map(self, origin_lat: float, origin_lon: float, retrieved_places: list[RetrievedPlace], output_path: Path | None = None) -> Path:
        import folium
        ensure_legacy_data_dirs(self.config)
        output = output_path or self.config.latest_map_path
        fmap = folium.Map(location=[origin_lat, origin_lon], zoom_start=15)
        folium.Marker([origin_lat, origin_lon], popup="Search origin", icon=folium.Icon(color="blue", icon="home")).add_to(fmap)
        for item in retrieved_places:
            record = item.record
            popup = f"<b>{record.name}</b><br>Category: {record.category}<br>Distance: {round(record.distance_to_origin_m)}m<br>Themes: {', '.join(record.themes) if record.themes else 'none'}<br>Contextual evidence: {', '.join(item.contextual_evidence) if item.contextual_evidence else 'none'}<br>Spatial relations: {relation_summary(record) or 'none'}<br>Match: {'strong' if item.is_strong_match else 'weak'}<br>Combined score: {item.combined_score}"
            folium.Marker([record.latitude, record.longitude], popup=popup, icon=folium.Icon(color="green", icon="info-sign")).add_to(fmap)
        output.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(output))
        return output