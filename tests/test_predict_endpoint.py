import sys
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

# Ensure repo root is on sys.path so `import app` works during pytest run.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_predict_fallback_or_model(monkeypatch):
    """Call `/predict` and assert a 200 JSON response with 'predictions'."""
    model_path = str(ROOT / "model_store" / "linear_regression.onnx")
    monkeypatch.setenv("MODEL_PATH", model_path)

    payload = {"input": [1, 2, 3]}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, dict)
    assert "predictions" in data


def test_predict_shape_handling(monkeypatch):
    """Verify that 1-D numeric lists are handled by the runtime for ONNX.

    If an ONNX model is present and expects shape (N,1), this test ensures
    the runtime accepts a 1-D list and returns a prediction rather than
    raising a 4xx/5xx.
    """
    model_path = str(ROOT / "model_store" / "linear_regression.onnx")
    monkeypatch.setenv("MODEL_PATH", model_path)

    payload = {"input": [1.0, 2.0, 3.0]}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "predictions" in data


def test_predict_returns_503_without_model_path(monkeypatch):
    """Verify `/predict` is unavailable when MODEL_PATH is not configured."""
    monkeypatch.delenv("MODEL_PATH", raising=False)

    payload = {"input": [1.0, 2.0, 3.0]}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert "model not loaded" in response.json()["detail"].lower()
