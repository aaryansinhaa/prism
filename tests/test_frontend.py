"""Tests for frontend UI - Updated for HTMX-based implementation."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_upload_ui_loads():
    """Test that the dashboard loads with PRISM branding and sidebar."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "PRISM" in response.text
    assert "Control Center" in response.text or "Model Control Center" in response.text
    assert "📊" in response.text  # Dashboard emoji


def test_predict_ui_loads():
    """Test that the prediction UI page loads."""
    with TestClient(app) as client:
        response = client.get("/predict")

    assert response.status_code == 200
    assert "PRISM" in response.text
    assert "🔮 Make Predictions" in response.text or "Send input data" in response.text
    assert 'hx-post="/predict-result"' in response.text  # HTMX form
    assert "📊 Dashboard" not in response.text
    assert "📤 Upload Model" not in response.text


def test_predict_ui_has_model_id_param():
    """Test that prediction UI page includes JavaScript for model_id URL parameter."""
    with TestClient(app) as client:
        response = client.get("/predict?model_id=test123")

    assert response.status_code == 200
    # Check that JavaScript code for parsing URL is present
    assert "URLSearchParams" in response.text or "model_id" in response.text
    assert 'hx-post="/predict-result"' in response.text


def test_upload_model_page_loads():
    """Test that the upload model page loads from sidebar."""
    with TestClient(app) as client:
        response = client.get("/upload-model")

    assert response.status_code == 200
    assert "Upload New Model" in response.text or "📤" in response.text
    assert 'hx-post="/api/upload-and-run-ui"' in response.text


def test_model_logs_page_loads():
    """Test that the model logs page loads from sidebar."""
    with TestClient(app) as client:
        response = client.get("/model-logs")

    assert response.status_code == 200
    assert "📋" in response.text or "Model Logs" in response.text


def test_dashboard_shows_no_models_when_empty():
    """Test that dashboard shows empty state when no models deployed."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    # Should show empty state or message about no models
    assert "No Models Deployed" in response.text or "Upload Model" in response.text
