from __future__ import annotations

from geoinsight.processing.spatial_features import haversine_distance_m
from geoinsight.schemas import PlaceRecord


class SpatialIndex:
    def __init__(self, records: list[PlaceRecord]):
        self.records = records

    def within_radius(self, lat: float, lon: float, radius_m: float) -> list[PlaceRecord]:
        return [
            record
            for record in self.records
            if haversine_distance_m(lat, lon, record.latitude, record.longitude) <= radius_m
        ]
