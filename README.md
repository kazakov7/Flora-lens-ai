# Flora Lens AI

**Plant species recognition from photos and live camera captures, powered by a real PlantCLEF 2024 model.**

[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PlantCLEF](https://img.shields.io/badge/Model-PlantCLEF%202024-315C45)](https://huggingface.co/gerald29/plantclef2024)

Flora Lens AI is a full-stack computer vision application that identifies plants from uploaded images or browser camera frames. It runs genuine DINOv2-based inference and returns five ranked scientific-name predictions with model scores and inference time.

> Plant identification is an AI estimate. Do not use the result to decide whether a plant is edible, medicinal, toxic, or safe to touch.

## Highlights

- Real pretrained PlantCLEF inference — no generated or placeholder predictions
- 7,806 supported plant species
- Top-5 scientific names, PlantNet species IDs, scores, and inference timing
- Drag-and-drop upload with image preview
- Manual camera capture and controlled 8-second auto-recognition
- Request locking that prevents overlapping camera inference
- JPG, PNG, and WebP validation with a 10 MB upload limit
- EXIF orientation correction and RGB conversion
- Responsive, accessible botanical interface
- CPU inference by default, with optional CUDA or Apple MPS support
- Model loaded once and reused across requests

## Application flow

```text
Upload or camera capture
          │
          ▼
React + TypeScript frontend
          │  multipart image
          ▼
FastAPI /predict endpoint
          │
          ▼
Official image preprocessing → DINOv2 PlantCLEF model
          │
          ▼
Top-5 species → scientific-name mapping → UI results
```

## Technology

| Layer | Technology |
| --- | --- |
| Frontend | React 19, TypeScript, Vite |
| API | Python 3.11, FastAPI, Uvicorn |
| AI inference | PyTorch, Hugging Face Transformers |
| Image handling | Pillow |
| Tests | Pytest, FastAPI TestClient |

## Model

| Property | Value |
| --- | --- |
| Hugging Face ID | [`gerald29/plantclef2024`](https://huggingface.co/gerald29/plantclef2024) |
| Architecture | DINOv2 image classifier |
| Parameters | 98,578,302 |
| Output classes | 7,806 |
| Input | 224 × 224 center crop after resizing the shortest edge to 256 px |
| Normalization | ImageNet mean and standard deviation |
| Default device | CPU |

The Hugging Face configuration maps model outputs to PlantNet species IDs. [`species_labels.json`](backend/app/species_labels.json) maps the same output positions to scientific names. The mapping order was checked against the official PlantCLEF example: output index `2` corresponds to species ID `1355870`, **Crepis foetida L.**

Sources:

- [Hugging Face model repository](https://huggingface.co/gerald29/plantclef2024)
- [Official PlantCLEF 2024 model record](https://zenodo.org/records/10848263)
- [DINOv2 repository](https://github.com/facebookresearch/dinov2)

The application downloads the approximately 394 MB Transformers model on first startup and stores it in the Hugging Face cache. It does not download the 2.3 GB original Zenodo archive.

## Project structure

```text
Flora-lens-ai/
├── backend/
│   ├── app/
│   │   ├── inference.py          # Model loading and Top-5 inference
│   │   ├── main.py               # FastAPI routes and image validation
│   │   └── species_labels.json   # 7,806 scientific names
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Upload, camera, and result experience
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.ts
├── tests/
│   └── test_api.py
├── AGENTS.md
└── README.md
```

## Requirements

- Python 3.11 or newer
- Node.js 20 or newer
- npm
- Approximately 1 GB of available disk space for Python packages and model cache

## Quick start

Both the backend and frontend must be running.

### 1. Install the backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### 2. Start the API

Keep this terminal open:

```bash
source .venv/bin/activate
HF_HOME="$PWD/.cache/huggingface" \
  uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The first start downloads and initializes the model. Wait for:

```text
Application startup complete.
```

### 3. Install and start the frontend

Open a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, normally [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite may select another port, such as `5174`, when `5173` is already occupied.

Camera access requires browser permission and a secure context. Localhost is treated as a secure context by current mainstream browsers.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `FLORA_DEVICE` | `cpu` | Use `cpu`, `cuda`, `mps`, or `auto` |
| `HF_HOME` | Hugging Face default | Model cache directory |
| `VITE_API_URL` | `/api` | Frontend API base URL; `/api` uses the local Vite proxy |

Example automatic hardware selection:

```bash
FLORA_DEVICE=auto HF_HOME="$PWD/.cache/huggingface" \
  uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## API

Interactive OpenAPI documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) while the backend is running.

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model": "gerald29/plantclef2024",
  "model_ready": true,
  "device": "cpu"
}
```

### Plant prediction

```http
POST /predict
Content-Type: multipart/form-data
```

The multipart field must be named `file`.

```bash
curl -X POST \
  -F "file=@plant.jpg" \
  http://127.0.0.1:8000/predict
```

Example response shape:

```json
{
  "model": "gerald29/plantclef2024",
  "predictions": [
    {
      "rank": 1,
      "scientific_name": "Orchis italica Poir.",
      "species_id": "1361678",
      "score": 0.62
    }
  ],
  "inference_time_ms": 984.2
}
```

Scores are softmax model estimates, not guaranteed or calibrated identification accuracy.

## Testing

Run backend tests from the repository root:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m pytest -q
```

Run the TypeScript check and production build:

```bash
cd frontend
npm run build
```

Verified locally:

- Backend API tests: **5 passed**
- Frontend TypeScript and production build: passed
- Real model initialization: passed
- Real Top-5 image inference: passed
- Invalid image rejection: passed (`400`)
- Oversized upload rejection: passed (`413`)
- Vite-to-FastAPI proxy integration: passed

## Troubleshooting

### “The recognition API is offline”

The frontend is running, but FastAPI is not reachable. Start the backend from the repository root and keep its terminal open:

```bash
source .venv/bin/activate
HF_HOME="$PWD/.cache/huggingface" \
  uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Confirm it is ready:

```bash
curl http://127.0.0.1:8000/health
```

### The first backend startup is slow

The model must be downloaded and initialized once. Later starts reuse the local cache.

### Camera access fails

- Allow camera permission in the browser.
- Use localhost or HTTPS.
- Close other applications that may be using the camera.
- Upload an image instead if the device does not expose `getUserMedia`.

## Limitations

- The training flora focuses on southwestern Europe, so species outside that distribution may be misidentified.
- Similar species and hybrids may require details that are not visible in a single image.
- Poor lighting, blur, occlusion, or multiple plants can reduce prediction quality.
- The application does not provide medical, toxicological, or food-safety advice.
- CPU inference latency depends on the host machine.

## Licensing and attribution

The Hugging Face repository declares an MIT license. The upstream DINOv2 code and weights are distributed under Apache-2.0. Review the official PlantCLEF Zenodo record and all upstream terms before redistributing model weights or claiming commercial-use rights.

This repository does not currently include a license for the Flora Lens AI application source code.
