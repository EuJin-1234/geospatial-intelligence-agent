from __future__ import annotations

from collections import Counter
from typing import Any

from geoinsight.schemas import PlaceRecord


def profile_place_records(records: list[PlaceRecord]) -> dict[str, Any]:
    names = [record.name.strip().lower() for record in records if record.name]
    categories = Counter(record.category for record in records)
    themes = Counter(theme for record in records for theme in record.themes)
    return {
        "total_records": len(records),
        "named_records": sum(1 for record in records if not record.name.lower().startswith("unnamed")),
        "unnamed_records": sum(1 for record in records if record.name.lower().startswith("unnamed")),
        "category_counts": dict(categories.most_common()),
        "theme_counts": dict(themes.most_common()),
        "duplicate_name_count": sum(count - 1 for count in Counter(names).values() if count > 1),
        "missing_value_counts": {
            "name": sum(1 for record in records if not record.name),
            "category": sum(1 for record in records if not record.category),
            "themes": sum(1 for record in records if not record.themes),
            "llm_description": sum(1 for record in records if not record.llm_description),
        },
    }
