# Local Docker Run

Use this to test the Azure-style free demo mode locally. The container runs FastAPI and uses the deterministic template provider, so it does not require Ollama or Azure credentials.

```bash
docker build -t geoinsight-agent .
docker run --env-file deployment/azure-env.example -p 8000:8000 geoinsight-agent
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

The Dockerfile copies the current `data/reports` and `data/features` demo artifacts into the image. The FAISS index is ignored for public repository safety, so `/ready` may report the vector index as missing in a fresh container unless you regenerate or mount index artifacts.

If artifacts are missing or stale, rebuild them before local runs:

```bash
python -m geoinsight.cli run-eda
python -m geoinsight.cli data-quality
python -m geoinsight.cli build-features
python -m geoinsight.cli build-index
```

Ollama is not run inside this container. Use `LLM_PROVIDER=template` for the no-cost demo path, or configure Azure OpenAI separately with credentials.
