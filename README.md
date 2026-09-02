# OpenScaner

![OpenScaner banner](assets/openscaner-banner.png)

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![CPU-only inference](https://img.shields.io/badge/Inference-CPU--only-00A6D6)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white)
![SHA-256 verified models](https://img.shields.io/badge/Models-SHA--256%20verified-4C566A)
[![Chinese README](https://img.shields.io/badge/Docs-ZH--CN-EF4444)](README_CN.md)

Vibed coded with Codex.

OpenScaner evaluates CPU-only document-boundary candidates from independent
candidate artifacts.

## Quick start

Install dependencies and download the third-party model weights:

```bash
uv sync --group ml
./scripts/download_models.sh
```

The repository keeps OpenScaner-trained weights in `models/`, but third-party
pretrained weights are ignored by git and fetched on demand. The download script
reads `models/manifest.json`, writes into `models/`, and verifies every fetched
file against its pinned SHA-256 before use.

## Document-boundary candidates

`docaligner_prompted_mobile_sam` is a fully automatic, source-image-only
candidate. It derives a coarse quadrilateral from DocAligner, generates bounded
MobileSAM prompts from that result, and either selects a verified mask or uses
the DocAligner quadrilateral as its deterministic fallback. It accepts no manual
prompts, ROI values, reference corners, or target-specific coordinates at
runtime.

Its policy, DocAligner checkpoint, and MobileSAM checkpoint are all verified by
their frozen filename, byte size, and SHA-256 identity before inference.

## OpenScaner service

The service exposes a browser UI and `/api/v1` endpoints for single
image scans and batch jobs. It uses FastAPI, SQLite, and local filesystem
storage, with one API process and one worker process sharing the same `data/`
directory.

Install dependencies:

```bash
uv sync --group ml
```

Run the Web UI and API locally:

```bash
OPENSCANER_AUTH_DISABLED=true \
OPENSCANER_STORAGE_ROOT=/tmp/openscaner-service \
uv run uvicorn openscaner.service.api:create_app --factory --host 127.0.0.1 --port 8000
```

Run the batch worker in a second terminal:

```bash
OPENSCANER_AUTH_DISABLED=true \
OPENSCANER_STORAGE_ROOT=/tmp/openscaner-service \
uv run python -m openscaner.service.worker
```

Open `http://127.0.0.1:8000/` for the Web UI. With authentication enabled,
enter either an API key or the configured web password in the credential field.

Docker deployment:

```bash
cp .env.example .env
docker compose up --build
```

Edit `.env` before deploying. Production should set `OPENSCANER_WEB_PASSWORD`
and `OPENSCANER_API_KEYS`, and should leave `OPENSCANER_AUTH_DISABLED=false`.

Synchronous single-image API:

```bash
curl -H "X-API-Key: change-me-api-key" \
  -F "image=@page.jpg" \
  http://localhost:8000/api/v1/scan
```

Asynchronous batch API:

```bash
curl -H "X-API-Key: change-me-api-key" \
  -F "images=@page1.jpg" -F "images=@page2.jpg" \
  http://localhost:8000/api/v1/jobs > job.json

JOB_ID="$(python - <<'PY'
import json
print(json.load(open("job.json"))["id"])
PY
)"

curl -H "X-API-Key: change-me-api-key" \
  "http://localhost:8000/api/v1/jobs/${JOB_ID}"

curl -H "X-API-Key: change-me-api-key" \
  -o results.zip \
  "http://localhost:8000/api/v1/jobs/${JOB_ID}/download.zip"
```

Retention and storage are controlled by:

- `OPENSCANER_STORAGE_ROOT`, default `data`
- `OPENSCANER_RETENTION_DAYS`, default `30`
- `OPENSCANER_KEEP_INPUTS`, default `true`
- `OPENSCANER_MAX_UPLOAD_MB`, default `50`
- `OPENSCANER_MAX_BATCH_ITEMS`, default `500`
