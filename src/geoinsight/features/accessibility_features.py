from __future__ import annotations


def build_accessibility_features(feature_rows: list[dict]) -> list[dict]:
    rows = []
    for row in feature_rows:
        transport = _within(row.get("nearest_transport_distance_m"), 400)
        food = _within(row.get("nearest_food_distance_m"), 400)
        green = _within(row.get("nearest_green_space_distance_m"), 500)
        study = _within(row.get("nearest_study_place_distance_m"), 400)
        score = (
            0.35 * float(transport)
            + 0.25 * float(food)
            + 0.2 * float(green)
            + 0.2 * float(study)
        )
        rows.append(
            {
                "place_id": row["place_id"],
                "has_transport_nearby": transport,
                "has_food_nearby": food,
                "has_green_space_nearby": green,
                "has_study_place_nearby": study,
                "walkability_proxy_score": round(score, 3),
            }
        )
    return rows


def _within(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold
