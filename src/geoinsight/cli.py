from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from geoinsight.config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_M
from geoinsight.indexing.embedder import export_sentence_transformer_to_onnx
from geoinsight.data_quality.validation import validate_place_records
from geoinsight.eda.report_builder import build_eda_report, load_records
from geoinsight.evaluation.benchmark_queries import run_benchmark
from geoinsight.features.feature_pipeline import build_place_features
from geoinsight.config import load_config
from geoinsight.logging_config import configure_logging
from geoinsight.pipeline.build_dataset import build_dataset as run_build_dataset
from geoinsight.pipeline.build_index import build_index as run_build_index
from geoinsight.schemas import DatasetBuildOptions, QueryOptions
from geoinsight.services.query_service import run_agent_query

app = typer.Typer(help="GeoInsight Agent CLI")
console = Console()


def _render_response(response) -> None:
    console.print(f"[bold]Query:[/bold] {response.query}")
    console.print(f"[bold]Answer:[/bold] {response.answer}")
    if response.map_path:
        console.print(f"[bold]Map:[/bold] {response.map_path}")
    if response.latency_ms is not None:
        console.print(f"[bold]Latency:[/bold] {response.latency_ms:.1f} ms")

    if not response.retrieved_places:
        console.print("[yellow]No places were retrieved.[/yellow]")
        return

    table = Table(title="Retrieved places")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Distance")
    table.add_column("Score")
    table.add_column("Reason")
    for item in response.retrieved_places:
        record = item.record
        table.add_row(
            record.name,
            record.category,
            f"{record.distance_to_origin_m:.0f} m",
            f"{item.combined_score:.3f}",
            item.match_reason,
        )
    console.print(table)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging(verbose)


@app.command("build-dataset")
def build_dataset(
    lat: float = typer.Option(DEFAULT_LAT, "--lat"),
    lon: float = typer.Option(DEFAULT_LON, "--lon"),
    radius: int = typer.Option(DEFAULT_RADIUS_M, "--radius"),
    force_refresh: bool = typer.Option(False, "--force-refresh"),
) -> None:
    report = run_build_dataset(
        DatasetBuildOptions(lat=lat, lon=lon, radius_m=radius, force_refresh=force_refresh)
    )
    console.print(f"[bold green]Built {report.records_built} place records.[/bold green]")


@app.command("run-eda")
def run_eda() -> None:
    report = build_eda_report(load_records())
    console.print(
        f"[bold green]EDA generated for {report.dataset_overview.get('cleaned_places', 0)} places.[/bold green]"
    )


@app.command("build-features")
def build_features() -> None:
    rows = build_place_features(load_records())
    console.print(f"[bold green]Built engineered features for {len(rows)} places.[/bold green]")


@app.command("build-index")
def build_index(
    lat: float = typer.Option(DEFAULT_LAT, "--lat"),
    lon: float = typer.Option(DEFAULT_LON, "--lon"),
    radius: int = typer.Option(DEFAULT_RADIUS_M, "--radius"),
    force_refresh: bool = typer.Option(False, "--force-refresh"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    onnx_model_dir: str | None = typer.Option(None, "--onnx-model-dir"),
) -> None:
    count = run_build_index(lat, lon, radius, force_refresh, embedding_model, embedding_backend, onnx_model_dir)
    console.print(f"[bold green]Built vector index for {count} place records.[/bold green]")


@app.command("export-onnx-embeddings")
def export_onnx_embeddings(
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    output_dir: str | None = typer.Option(None, "--output-dir"),
) -> None:
    config = load_config()
    output = export_sentence_transformer_to_onnx(
        embedding_model or config.legacy.embedding_model_name,
        output_dir or config.legacy.onnx_embedding_model_dir,
    )
    console.print(f"[bold green]Exported ONNX embedding model to {output}[/bold green]")


@app.command("query")
def query(
    text: Annotated[str, typer.Argument(help="Natural-language geospatial query")],
    top_k: int = typer.Option(5, "--top-k"),
    max_distance: int = typer.Option(1500, "--max-distance"),
    theme: list[str] = typer.Option([], "--theme"),
    map_requested: bool = typer.Option(False, "--map"),
) -> None:
    response = run_agent_query(
        text,
        QueryOptions(
            top_k=top_k,
            max_distance_m=max_distance,
            requested_themes=theme,
            map_requested=map_requested,
        ),
    )
    _render_response(response)


@app.command("ask")
def ask(text: Annotated[str, typer.Argument(help="Question for GeoInsight Agent")]) -> None:
    _render_response(run_agent_query(text, QueryOptions()))


@app.command("evaluate")
def evaluate() -> None:
    output = run_benchmark()
    console.print(f"[bold green]Evaluation report written to {output}[/bold green]")


@app.command("data-quality")
def data_quality() -> None:
    report = validate_place_records(load_records())
    console.print(f"[bold green]Data quality warnings: {report.warning_count}[/bold green]")


@app.command("serve-api")
def serve_api(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool | None = typer.Option(None, "--reload/--no-reload"),
) -> None:
    import uvicorn

    config = load_config()
    uvicorn.run(
        "geoinsight.api.app:app",
        host=host or config.api_host,
        port=port or config.api_port,
        reload=config.api_reload if reload is None else reload,
    )


if __name__ == "__main__":
    app()