# Flora Lens AI

Flora Lens AI is a local, full-stack plant recognition application. It accepts an uploaded photo or a browser camera capture and returns the five most likely scientific names from a genuine PlantCLEF 2024 image-classification model.

## Features

- Real pretrained DINOv2/PlantCLEF inference — no placeholder predictions
- Top-5 scientific-name results with model scores and inference time
- Drag-and-drop upload with preview, validation, and a 10 MB limit
- Camera capture with manual scanning and a non-overlapping 8-second auto-scan
- Responsive, accessible botanical interface
- Consistent API errors for invalid, unsupported, and oversized files
- One reusable model instance loaded during API startup

## Technology

- React 19, TypeScript, Vite
- FastAPI, Python 3.11
- PyTorch, Hugging Face Transformers
- Pillow

## Model

- Hugging Face model: [`gerald29/plantclef2024`](https://huggingface.co/gerald29/plantclef2024)
- Official PlantCLEF reference: [Zenodo record 10848263](https://zenodo.org/records/10848263)
- Architecture: DINOv2 image classifier
- Parameters: 98,578,302
- Supported output classes: **7,806**
- Input preprocessing: shortest edge 256 px, center crop 224 × 224, ImageNet normalization

The Hugging Face configuration maps output indices to PlantNet species IDs. `backend/app/species_labels.json` resolves the same 7,806 output positions to scientific names. The ordering was checked against the official PlantCLEF example: output index 2 maps to species ID `1355870`, `Crepis foetida L.`

The selected Hugging Face repository declares an MIT license. The upstream DINOv2 code and weights are Apache-2.0. The official PlantCLEF Zenodo record should be reviewed directly before redistribution or commercial use because its displayed rights metadata does not currently state a license. No commercial-use claim is made here.

## Install

Requirements: Python 3.11+, Node.js 20+, npm, and approximately 1 GB of free disk space for dependencies and the model cache.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt

cd frontend
npm install
cd ..
```

The model downloads from Hugging Face on the first backend start and is cached locally. The large 2.3 GB original Zenodo archive is not needed.

## Run

Start the API from the project root:

```bash
source .venv/bin/activate
HF_HOME="$PWD/.cache/huggingface" uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. Camera access requires browser permission and a secure context; localhost is accepted by modern browsers.

CPU inference is the default. Supported acceleration can be selected explicitly with `FLORA_DEVICE=auto`, `FLORA_DEVICE=mps`, or `FLORA_DEVICE=cuda` before the backend command.

## API

### `GET /health`

Returns application status, model identifier, readiness, and active device.

### `POST /predict`

Send multipart form data with one `file` field containing a JPG, PNG, or WebP image up to 10 MB.

```bash
curl -X POST -F "file=@plant.jpg" http://127.0.0.1:8000/predict
```

The response contains the actual model identifier, five ranked predictions, PlantNet species IDs, softmax scores, and inference time.

## Verification

Completed in the provided macOS x86_64 environment:

- Real model load: passed — 7,806 outputs and 98,578,302 parameters
- Real image inference: passed — Top-5 scientific-name predictions returned
- Backend tests: `5 passed`
- Frontend TypeScript and production build: passed
- Live `/health`: passed (`200`)
- Live `/predict` with a real image: passed (`200`)
- Invalid image rejection: passed (`400`)
- Oversized upload rejection: passed (`413`)
- Vite-to-FastAPI proxy: passed (`200`)

Run the automated checks with:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m pytest -q
cd frontend && npm run build
```

## Limitations

- Scores are model estimates, not calibrated probabilities or guaranteed identifications.
- The training flora focuses on southwestern Europe, so plants outside that distribution can be misidentified.
- A single photo may omit diagnostic traits needed to distinguish similar species or hybrids.
- Camera permission and capture behavior require testing on the target browser/device.
- Do not use predictions to decide whether a plant is edible, medicinal, toxic, or safe to touch.

