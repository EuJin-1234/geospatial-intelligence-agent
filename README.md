# GeoInsight Agent

GeoInsight Agent is a geospatial AI system that turns OpenStreetMap data into validated spatial features, semantic search indexes, and evidence-grounded AI responses.

## Why this project matters

Real-world AI systems often need to reason about places, amenities, accessibility, and local context. Raw geospatial data is noisy and difficult for LLMs to use directly. GeoInsight Agent bridges that gap by turning OpenStreetMap data into structured features, retrieval indexes, and evidence-grounded responses for location-intelligence tasks.

## Key Features

**Data pipeline**
- OpenStreetMap ingestion
- Data quality validation
- Exploratory data analysis
- Spatial, density, accessibility, and semantic feature engineering

**Retrieval and reasoning**
- FAISS vector indexing
- Hybrid spatial-semantic retrieval
- Config-driven thematic mapping
- Deterministic reranking with evidence
- LangGraph-based query workflow

**Interfaces and deployment**
- CLI workflows
- Streamlit dashboard
- FastAPI backend
- Docker support
- Local Ollama mode
- Cloud-safe template provider mode
- Azure Container Apps deployment guide
- Offline test suite

## Architecture

```text
OpenStreetMap / GeoJSON
        |
Data Ingestion
        |
Cleaning + Data Quality
        |
EDA Reports
        |
Feature Engineering
        |
Spatial + Semantic Indexing
        |
Hybrid Retrieval + Reranking
        |
LangGraph Agent Workflow
        |
LLM Provider
        |
Service Layer
        |
CLI / Streamlit / FastAPI / Docker
```

## Quickstart

```bash
git clone <repo-url>
cd geospatial-intelligence-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Build local data pipeline

```bash
python -m geoinsight.cli build-dataset
python -m geoinsight.cli run-eda
python -m geoinsight.cli data-quality
python -m geoinsight.cli build-features
python -m geoinsight.cli build-index
```

## Run CLI query

```bash
python -m geoinsight.cli query "Find a quiet cafe near campus"
python -m geoinsight.cli ask "What are the most common amenity categories?"
```

## Run Streamlit dashboard

```bash
python -m streamlit run app.py
```

## Run FastAPI backend

```bash
python -m uvicorn geoinsight.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

```bash
docker build -t geoinsight-agent .
docker run -p 8000:8000 --env-file deployment/azure-env.example geoinsight-agent
```

## LLM provider modes

### Local Ollama mode

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

Use this for local open-source LLM experimentation.

### Cloud-safe template mode

```env
LLM_PROVIDER=template
```

Use this for free/low-cost cloud demos without paid LLM calls. This mode creates deterministic evidence summaries from retrieved records; it is not a real LLM.

### Optional Azure mode

```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Azure OpenAI may incur usage-based costs. The Azure provider is structurally prepared, but generation still needs a real Azure SDK call before production use.

## API Endpoints

- `GET /health`
- `GET /ready`
- `POST /query`
- `GET /reports/eda`
- `GET /reports/data-quality`
- `GET /features/preview`
- `POST /build/dataset`
- `POST /build/eda`
- `POST /build/features`
- `POST /build/index`

## Example API query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Find study-friendly places near campus\",\"top_k\":5,\"generate_map\":true}"
```


## Optional Edge AI Embeddings with ONNX Runtime

Install optional edge dependencies:

```bash
pip install -e ".[edge]"
```

Export the embedding model:

```bash
python -m geoinsight.cli export-onnx-embeddings
```

Use the ONNX embedding backend when building the FAISS index:

```bash
python -m geoinsight.cli build-index --embedding-backend onnx
```

Equivalent environment configuration:

```env
EMBEDDING_BACKEND=onnx
ONNX_EMBEDDING_MODEL_DIR=data/index/embedding_onnx
```

If ONNX Runtime or the exported model is unavailable, the system logs a warning and falls back to the normal SentenceTransformer embedder. Latency reports include the configured embedding backend and embedding latency, making the retrieval layer easy to benchmark for edge deployments.

Future work for LLM inference: ONNX Runtime GenAI, TensorRT-LLM, llama.cpp, quantisation, and device-specific memory optimisation.

## Testing

```bash
pytest
```

## License

MIT License. See [LICENSE](LICENSE).