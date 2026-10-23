# Project

Flora Lens AI identifies plant species from uploaded images and camera frames using a pretrained PlantCLEF 2024 model.

# Tech Stack

- React, TypeScript, Vite
- Python, FastAPI
- PyTorch, Hugging Face Transformers
- Pillow; OpenCV only if necessary

# Architecture

- `frontend/` — React application
- `backend/` — FastAPI and AI inference
- `tests/` — Essential tests

# Rules

- Use a genuine pretrained PlantCLEF model.
- Never train or fine-tune models.
- Never generate fake AI predictions.
- Prefer simple, maintainable solutions.
- Keep components and inference logic modular.
- Load the model once, not per request.
- Validate uploaded images.
- Handle errors clearly.
- Avoid unnecessary dependencies and features.
- Run relevant tests after implementation.
- Never commit model weights, secrets, or environment files.
