# Azure Container Apps Deployment

This guide prepares a free/low-cost cloud demo of GeoInsight Agent on Azure Container Apps.

## Deployment Modes

- Free/low-cost demo: FastAPI plus saved reports/features artifacts, with `LLM_PROVIDER=template`. A vector index can be regenerated or mounted for full query readiness.
- Optional Azure AI mode: FastAPI plus Azure OpenAI, only when credentials are provided.
- Local development remains FastAPI, Streamlit, Ollama, and local FAISS.

Do not run Ollama inside this container. The template provider is deterministic and intended only for a safe cloud demo without paid model calls.

## Cost Note

Azure Container Apps includes a monthly free grant, but Azure Container Registry, Log Analytics, traffic, image storage, and usage beyond free grants may incur costs. Delete resources after a demo if they are not needed.

## Option A: Azure Portal

1. Build a Docker image locally or through a future CI workflow.
2. Push the image to Azure Container Registry or another registry supported by Azure Container Apps.
3. Create a Container Apps environment.
4. Create a Container App using the pushed image.
5. Set environment variables from `deployment/azure-env.example`.
6. Expose target port `8000` with external ingress.
7. Test:

```bash
curl https://<your-container-app-url>/health
curl https://<your-container-app-url>/ready
```

API docs are available at:

```text
https://<your-container-app-url>/docs
```

## Option B: Azure CLI

Replace placeholders such as `<unique_acr_name>` with your own globally unique values.

```bash
az login

az group create \
  --name geoinsight-rg \
  --location uksouth

az acr create \
  --resource-group geoinsight-rg \
  --name <unique_acr_name> \
  --sku Basic

az acr login --name <unique_acr_name>

docker build -t <unique_acr_name>.azurecr.io/geoinsight-agent:latest .
docker push <unique_acr_name>.azurecr.io/geoinsight-agent:latest

az containerapp env create \
  --name geoinsight-env \
  --resource-group geoinsight-rg \
  --location uksouth

az containerapp create \
  --name geoinsight-api \
  --resource-group geoinsight-rg \
  --environment geoinsight-env \
  --image <unique_acr_name>.azurecr.io/geoinsight-agent:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars LLM_PROVIDER=template ENVIRONMENT=production LOG_LEVEL=INFO API_HOST=0.0.0.0 API_PORT=8000
```

After deployment, get the public URL from the Azure Portal or CLI output and test:

```bash
curl https://<your-container-app-url>/health
curl https://<your-container-app-url>/ready
```

## Optional Azure OpenAI Mode

Only enable this when you have a deployed Azure OpenAI model and understand usage costs:

```bash
az containerapp update \
  --name geoinsight-api \
  --resource-group geoinsight-rg \
  --set-env-vars \
    LLM_PROVIDER=azure \
    AZURE_OPENAI_ENDPOINT=<endpoint> \
    AZURE_OPENAI_DEPLOYMENT=<deployment> \
    AZURE_OPENAI_API_KEY=<api_key> \
    AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

The current Azure provider is a placeholder structure. A real Azure SDK call still needs to be implemented before Azure OpenAI generation is available.

## Cleanup

Delete all demo resources when finished:

```bash
az group delete --name geoinsight-rg --yes --no-wait
```
