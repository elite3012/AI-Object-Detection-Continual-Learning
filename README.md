# PestScope IP102

PestScope is a small ML system for pest-image triage on a reviewed subset of IP102. The project is built around one idea: a classifier should be trained, evaluated, versioned, and allowed to say "uncertain" instead of forcing a confident label.

The main model is `PestNet-S`, a compact residual CNN trained from scratch. The app serves a versioned model bundle through FastAPI and shows top-k predictions, confidence, supported classes, and offline review capture.

## Current Scope

- 12 reviewed IP102 pest classes with scientific names and Vietnamese display names.
- Manifest-based training from official IP102 splits.
- Custom CNN and simple CNN baseline code.
- Model bundle with weights, preprocessing, class map, metrics, and artifact hash.
- FastAPI inference API and a browser workspace.
- Docker runtime that starts even before a promoted model is mounted, using a clearly marked demo fallback.

IP102 is academic-use data. Images and training artifacts are not committed to this repository.

## Run The App

```powershell
python -m pip install -r requirements-dev.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

If `artifacts/models/pestnet_s_latest` does not exist, the API creates an untrained demo bundle so the UI and endpoints can be tested. Do not report demo fallback metrics as model performance.

## Train A Smoke Bundle

After Section 2 data artifacts exist:

```powershell
python scripts\train_pestnet.py `
  --max-epochs 1 `
  --limit-train-per-class 2 `
  --limit-val-per-class 1 `
  --device cpu `
  --bundle-dir artifacts\models\pestnet_s_smoke
```

Train the configured experiment:

```powershell
python scripts\train_pestnet.py --config configs\train\pestnet_s.yaml --device cpu
```

The default output bundle is `artifacts/models/pestnet_s_latest`.

## Docker

```powershell
docker compose up --build
```

The container exposes:

- Web app: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`
- Readiness: `http://localhost:8000/api/v1/health/ready`

Mount or copy a trained bundle into `/app/artifacts/models/pestnet_s_latest` for real inference. Without it, the container runs the marked demo fallback.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health/live` | Process liveness |
| `GET` | `/api/v1/health/ready` | Model readiness |
| `GET` | `/api/v1/model` | Model card, class map, preprocessing |
| `GET` | `/api/v1/examples` | Demo image metadata and attribution |
| `POST` | `/api/v1/predictions` | Predict one uploaded image |
| `POST` | `/api/v1/reviews` | Store offline human feedback |

## Verification

```powershell
python -m ruff check src\pestscope api.py scripts tests --no-cache
python -m ruff format --check src\pestscope api.py scripts tests
python -m pytest -q
docker compose config --quiet
```

The detailed design and section gates are tracked in `DESIGN_REPORT_IP102.md`.
