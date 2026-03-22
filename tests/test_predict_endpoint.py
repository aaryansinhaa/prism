import json
import time
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure repo root is on sys.path so `import app` works during pytest run.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_predict_fallback_or_model():
    """Call `/predict` and assert a 200 JSON response with 'predictions'.

    The repository may register a real ONNX/sklearn model or the fallback
    development model. Either way, the endpoint should respond with
    a JSON object containing a `predictions` key.
    """
    payload = {"input": [1, 2, 3]}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, dict)
    assert "predictions" in data


def test_predict_shape_handling():
    """Verify that 1-D numeric lists are handled by the runtime for ONNX.

    If an ONNX model is present and expects shape (N,1), this test ensures
    the runtime accepts a 1-D list and returns a prediction rather than
    raising a 4xx/5xx.
    """
    payload = {"input": [1.0, 2.0, 3.0]}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "predictions" in data
