from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "gerald29/plantclef2024"
LABELS_PATH = Path(__file__).with_name("species_labels.json")


class PlantClassifier:
    """Loads the PlantCLEF model once and provides thread-safe predictions."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.processor: Any | None = None
        self.model: Any | None = None
        self.labels: list[str] = []
        self.device = torch.device("cpu")
        self.ready = False
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _select_device(self) -> torch.device:
        requested = os.getenv("FLORA_DEVICE", "cpu").lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FLORA_DEVICE=cuda was requested, but CUDA is unavailable")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("FLORA_DEVICE=mps was requested, but MPS is unavailable")
        if requested not in {"cpu", "cuda", "mps"}:
            raise RuntimeError("FLORA_DEVICE must be cpu, cuda, mps, or auto")
        return torch.device(requested)

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=False)
            model = AutoModelForImageClassification.from_pretrained(self.model_id)
            if len(labels) != len(model.config.id2label):
                raise RuntimeError(
                    f"Label mapping has {len(labels)} entries, but model has "
                    f"{len(model.config.id2label)} outputs"
                )
            self.device = self._select_device()
            self.processor = processor
            self.model = model.to(self.device).eval()
            self.labels = labels
            self.ready = True

    def predict(self, image: Image.Image, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
        if not self.ready or self.model is None or self.processor is None:
            raise RuntimeError("Model is not ready")

        started = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with self._inference_lock, torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores, indices = torch.topk(probabilities, k=min(top_k, len(self.labels)))

        predictions = [
            {
                "rank": rank,
                "scientific_name": self.labels[index],
                "species_id": str(self.model.config.id2label[index]),
                "score": round(score, 6),
            }
            for rank, (score, index) in enumerate(
                zip(scores[0].tolist(), indices[0].tolist()), start=1
            )
        ]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return predictions, elapsed_ms

