from __future__ import annotations


def assign_themes(category: str, subcategory: str | None, tags: dict) -> list[str]:
    values = {str(category or "").lower(), str(subcategory or "").lower()}
    for key in ("amenity", "leisure", "shop", "public_transport", "tourism"):
        value = tags.get(key)
        if value not in (None, True, False):
            values.add(str(value).lower())
        elif value is True:
            values.add(key)
    themes: set[str] = set()
    if "library" in values:
        themes.update({"study", "quiet", "campus"})
    if "cafe" in values:
        themes.update({"coffee", "food", "social", "study"})
    if values.intersection({"restaurant", "fast_food"}):
        themes.update({"food", "social"})
    if values.intersection({"park", "garden"}):
        themes.update({"relax", "outdoor", "quiet"})
    if values.intersection({"bus_station", "public_transport", "stop_position", "platform"}):
        themes.add("transport")
    if values.intersection({"university", "school"}):
        themes.update({"campus", "study"})
    if "shop" in values or tags.get("shop"):
        themes.add("shopping")
    if "sports_centre" in values:
        themes.add("fitness")
    return sorted(themes)