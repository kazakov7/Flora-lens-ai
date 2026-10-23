from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app


class FakeClassifier:
    model_id = "test/plant-model"
    ready = True
    device = "cpu"

    def load(self) -> None:
        self.ready = True

    def predict(self, image: Image.Image, top_k: int = 5):
        assert image.mode == "RGB"
        predictions = [
            {
                "rank": rank,
                "scientific_name": f"Species {rank}",
                "species_id": str(1000 + rank),
                "score": round(0.8 / rank, 6),
            }
            for rank in range(1, top_k + 1)
        ]
        return predictions, 12.3


def image_bytes(image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(buffer, image_format)
    return buffer.getvalue()


def test_health_reports_model_ready() -> None:
    with TestClient(create_app(FakeClassifier())) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is True
    assert response.json()["model"] == "test/plant-model"


def test_predict_returns_top_five() -> None:
    with TestClient(create_app(FakeClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("leaf.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5
    assert body["predictions"][0]["scientific_name"] == "Species 1"
    assert body["inference_time_ms"] == 12.3


def test_predict_rejects_non_image() -> None:
    with TestClient(create_app(FakeClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("not-image.jpg", b"not an image", "image/jpeg")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "The file is not a valid image"


def test_predict_rejects_unsupported_image_format() -> None:
    with TestClient(create_app(FakeClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("plant.gif", image_bytes("GIF"), "image/gif")},
        )
    assert response.status_code == 415


def test_predict_rejects_oversized_upload() -> None:
    with TestClient(create_app(FakeClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("huge.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")},
        )
    assert response.status_code == 413

