from __future__ import annotations

from pathlib import Path

import pandas as pd

from geoinsight.config import load_config
from geoinsight.data_quality.validation import load_data_quality_report
from geoinsight.eda.report_builder import load_records
from geoinsight.features.feature_pipeline import load_place_features
from geoinsight.schemas import QueryOptions
from geoinsight.services.data_service import get_eda_summary
from geoinsight.services.query_service import run_agent_query as run_spatial_query
from geoinsight.schemas import QueryResponse, RetrievedPlace


def retrieved_places_dataframe(response: QueryResponse) -> pd.DataFrame:
    rows = []
    for item in sorted(response.retrieved_places, key=lambda result: result.combined_score, reverse=True):
        record = item.record
        rows.append(
            {
                "Name": record.name,
                "Category": record.category,
                "Semantic Score": item.semantic_score,
                "Spatial Score": item.spatial_score,
                "Theme Score": item.theme_score,
                "Contextual Score": item.contextual_score,
                "Combined Score": item.combined_score,
                "Match": "strong" if item.is_strong_match else "weak",
                "Themes": ", ".join(record.themes),
                "Distance": round(record.distance_to_origin_m),
                "Contextual Evidence": "; ".join(item.contextual_evidence) or item.match_reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GeoInsight Agent", layout="wide")
    _init_session_state(st)
    st.title("GeoInsight Agent")
    st.caption(
        "Geospatial AI analytics with data quality checks, engineered spatial features, "
        "hybrid retrieval, LangGraph orchestration, and evidence-grounded answers."
    )

    ask_tab, data_tab, feature_tab, architecture_tab = st.tabs(
        ["Ask GeoInsight", "Data Explorer", "Feature Explorer", "System Architecture"]
    )
    with ask_tab:
        _render_ask_tab(st)
    with data_tab:
        _render_data_tab(st)
    with feature_tab:
        _render_feature_tab(st)
    with architecture_tab:
        _render_architecture_tab(st)


def _render_ask_tab(st) -> None:
    left, right = st.columns([0.38, 0.62])
    with left:
        query = st.text_area("Query", value="Find a quiet cafe near campus where I can study")
        top_k = st.slider("Top-k", min_value=3, max_value=10, value=5)
        max_distance_m = st.slider("Max distance (metres)", 250, 3000, 1500, 50)
        requested_themes = st.multiselect(
            "Themes",
            options=["study", "food", "transport", "outdoor", "social", "coffee", "quiet"],
        )
        map_requested = st.checkbox("Generate map", value=True)
        run_clicked = st.button("Ask GeoInsight", type="primary")

    with right:
        if run_clicked:
            st.session_state["last_query"] = query
            st.session_state["last_query_error"] = None
            st.session_state["is_query_running"] = True
            try:
                with st.spinner("Running retrieval and grounded answer generation..."):
                    st.session_state["last_query_result"] = run_spatial_query(
                        query,
                        QueryOptions(
                            top_k=top_k,
                            max_distance_m=max_distance_m,
                            requested_themes=requested_themes,
                            map_requested=map_requested,
                        ),
                    )
            except Exception as exc:
                st.session_state["last_query_result"] = None
                st.session_state["last_query_error"] = str(exc)
            finally:
                st.session_state["is_query_running"] = False

        if st.session_state["last_query_error"]:
            st.error(st.session_state["last_query_error"])

        response = st.session_state["last_query_result"]
        if response is None:
            st.info("Ask a local area, recommendation, map, or EDA question.")
            return

        context = response.context
        st.write(
            {
                "Intent": context.intent if context else None,
                "Max distance": max_distance_m,
                "Fallback used": response.fallback_used,
            }
        )
        st.markdown(response.answer)
        st.dataframe(retrieved_places_dataframe(response), use_container_width=True, hide_index=True)
        if map_requested and response.map_path:
            _render_map(st, response)
        st.metric("Latency", f"{response.latency_ms:.2f} ms" if response.latency_ms else "n/a")


def _init_session_state(st) -> None:
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None
    if "last_query_result" not in st.session_state:
        st.session_state["last_query_result"] = None
    if "last_query_error" not in st.session_state:
        st.session_state["last_query_error"] = None
    if "is_query_running" not in st.session_state:
        st.session_state["is_query_running"] = False


def _render_data_tab(st) -> None:
    report = get_eda_summary()
    quality = load_data_quality_report()
    overview = report.get("dataset_overview", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Cleaned places", overview.get("cleaned_places", 0))
    c2.metric("Named places", overview.get("named_places", 0))
    c3.metric("Unnamed places", overview.get("unnamed_places", 0))
    st.subheader("Category Counts")
    st.dataframe(_dict_dataframe(overview.get("category_counts", {})), hide_index=True)
    st.subheader("Theme Counts")
    st.dataframe(
        _dict_dataframe(report.get("amenity_theme_overview", {}).get("top_themes", {})),
        hide_index=True,
    )
    if quality:
        st.subheader("Data Quality Warnings")
        st.write(quality.model_dump(mode="json"))
    _render_report_images(st)


def _render_feature_tab(st) -> None:
    features = load_place_features()
    if not features:
        st.info("No engineered feature file found yet. Run `python -m geoinsight.cli build-features`.")
        return
    frame = pd.DataFrame(features)
    categories = sorted(frame["category"].dropna().unique()) if "category" in frame else []
    selected = st.multiselect("Category filter", options=categories)
    if selected:
        frame = frame[frame["category"].isin(selected)]
    if "distance_band" in frame:
        bands = sorted(frame["distance_band"].dropna().unique())
        selected_bands = st.multiselect("Distance band filter", options=bands)
        if selected_bands:
            frame = frame[frame["distance_band"].isin(selected_bands)]
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(
        "Feature rows combine spatial distance, nearby amenities, density proxies, accessibility flags, and semantic themes."
    )


def _render_architecture_tab(st) -> None:
    st.code(
        "OpenStreetMap / GeoJSON\n"
        "  -> Raw Data Store\n"
        "  -> Data Validation + Cleaning\n"
        "  -> Exploratory Data Analysis\n"
        "  -> Feature Engineering\n"
        "  -> Spatial + Semantic Indexing\n"
        "  -> LangGraph Agent Workflow\n"
        "  -> Evidence-Grounded Answer\n"
        "  -> CLI / Streamlit UI\n"
        "  -> Future: FastAPI + Azure AI",
        language="text",
    )
    st.write(
        {
            "Current providers": ["OpenStreetMap", "FAISS", "Ollama"],
            "Prepared extensions": ["FastAPI service layer", "Azure OpenAI placeholder", "external spatial stores"],
            "Limitations": [
                "walking distance is approximated",
                "themes are deterministic config rules",
                "live opening hours, weather, traffic, and events are not included",
            ],
        }
    )


def _render_map(st, response: QueryResponse) -> None:
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        map_path = Path(response.map_path)
        st.info(f"Map saved to: {map_path}")
        st.caption("Install streamlit-folium to render the interactive map inside Streamlit.")
        return

    first_place = response.retrieved_places[0].record if response.retrieved_places else None
    centre = [first_place.latitude, first_place.longitude] if first_place else [50.9350, -1.3960]
    fmap = folium.Map(location=centre, zoom_start=15)
    for item in response.retrieved_places:
        record = item.record
        folium.Marker(
            [record.latitude, record.longitude],
            popup=(
                f"{record.name}<br>"
                f"{record.category}<br>"
                f"combined={item.combined_score}<br>"
                f"match={'strong' if item.is_strong_match else 'weak'}"
            ),
        ).add_to(fmap)
    st_folium(fmap, width=None, height=520)


def _dict_dataframe(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Name": key, "Count": value} for key, value in values.items()])


def _render_report_images(st) -> None:
    reports_dir = load_config().reports_dir
    for filename in ("category_counts.png", "theme_counts.png", "distance_distribution.png"):
        path = reports_dir / filename
        if path.exists():
            st.image(str(path))


if __name__ == "__main__":
    main()


