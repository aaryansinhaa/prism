"""Tests for frontend UI - Updated for HTMX-based implementation."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def test_upload_ui_loads():
    """Test that the upload UI page loads with PRISM branding."""
    with TestClient(app) as client:
        response = client.get("/")
    
    assert response.status_code == 200
    assert "PRISM" in response.text
    assert "🚀 PRISM" in response.text  # Logo emoji
    assert "Model Upload" in response.text or "Deploy ML models" in response.text
    assert 'hx-post="/api/upload-and-run-ui"' in response.text  # HTMX form


def test_predict_ui_loads():
    """Test that the prediction UI page loads."""
    with TestClient(app) as client:
        response = client.get("/predict")
    
    assert response.status_code == 200
    assert "PRISM" in response.text
    assert "🔮 Make Predictions" in response.text or "Send input data" in response.text
    assert 'hx-post="/predict-result"' in response.text  # HTMX form


def test_predict_ui_has_model_id_param():
    """Test that prediction UI page includes JavaScript for model_id URL parameter."""
    with TestClient(app) as client:
        response = client.get("/predict?model_id=test123")
    
    assert response.status_code == 200
    # Check that JavaScript code for parsing URL is present
    assert "URLSearchParams" in response.text or "model_id" in response.text
    assert 'hx-post="/predict-result"' in response.text

