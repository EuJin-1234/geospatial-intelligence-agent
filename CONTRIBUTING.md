# Contributing

Thanks for considering a contribution to GeoInsight Agent.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Branch Naming

Use short descriptive branches, for example:

```text
feature/api-readiness
fix/retrieval-ranking
docs/readme-update
```

## Tests

Run the full test suite before opening a pull request:

```bash
pytest
```

Tests should not require internet access, Ollama, Azure credentials, or paid services.

## Coding Style

- Keep changes focused.
- Prefer existing project patterns.
- Keep service logic out of API routes when possible.
- Add tests for behavior changes.
- Avoid committing generated cache files.

## Secrets

Do not commit secrets, API keys, `.env` files, Azure credentials, model weights, or local machine paths.

## Issues

When reporting issues, include:

- operating system
- Python version
- command run
- expected behavior
- actual behavior
- logs or error messages
