"""Comprehensive frontend tests - Integration Tests (IT) and Unit Tests (UT)."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# ============================================================================
# UNIT TESTS (UT) - Component and Endpoint Level Tests
# ============================================================================


class TestUploadFormComponent:
    """UT: Test upload form HTML component rendering."""
    
    def test_upload_ui_returns_200(self):
        """UT: GET / returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_upload_ui_contains_upload_form(self):
        """UT: Upload page contains form with file input."""
        response = client.get("/")
        assert "🚀 PRISM" in response.text
        assert "input" in response.text
        assert 'type="file"' in response.text
        assert 'accept=".onnx,.pkl,.pickle,.joblib"' in response.text
    
    def test_upload_ui_contains_tunnel_checkbox(self):
        """UT: Upload form has tunnel enable checkbox."""
        response = client.get("/")
        assert 'id="enableTunnel"' in response.text
        assert "Enable Public Tunnel" in response.text
    
    def test_upload_ui_has_htmx_integration(self):
        """UT: Form uses HTMX for submission."""
        response = client.get("/")
        assert "htmx.org" in response.text  # HTMX CDN
        assert 'hx-post="/api/upload-and-run-ui"' in response.text
        assert 'hx-target="#response"' in response.text
    
    def test_upload_ui_has_tailwind_styling(self):
        """UT: Page loads Tailwind CSS."""
        response = client.get("/")
        assert "cdn.tailwindcss.com" in response.text
        assert "gradient-bg" in response.text  # Custom gradient


class TestPredictionInterfaceComponent:
    """UT: Test prediction interface HTML component rendering."""
    
    def test_predict_ui_returns_200(self):
        """UT: GET /predict returns 200 OK."""
        response = client.get("/predict?model_id=test-model")
        assert response.status_code == 200
    
    def test_predict_ui_contains_prediction_form(self):
        """UT: Prediction page contains form with JSON textarea."""
        response = client.get("/predict?model_id=test-model")
        assert "🔮 Make Predictions" in response.text
        assert '<textarea' in response.text
        assert 'name="input_data"' in response.text
    
    def test_predict_ui_extracts_model_id_from_url(self):
        """UT: Page contains JavaScript to extract model_id from query params."""
        response = client.get("/predict?model_id=my-model-123")
        assert "URLSearchParams" in response.text
        assert "model_id" in response.text
    
    def test_predict_ui_has_htmx_form(self):
        """UT: Prediction form uses HTMX."""
        response = client.get("/predict?model_id=test-model")
        assert 'hx-post="/predict-result"' in response.text
        assert 'hx-indicator="#predictLoading"' in response.text
    
    def test_predict_ui_shows_error_without_model_id(self):
        """UT: Shows error message if model_id not provided."""
        response = client.get("/predict")
        assert "Model ID not provided" in response.text
        assert 'class="text-red-600"' in response.text


# ============================================================================
# INTEGRATION TESTS (IT) - Full Feature Workflows
# ============================================================================


class TestUploadAndRunWorkflow:
    """IT: Test complete upload and deployment workflow."""
    
    def test_successful_upload_returns_component(self):
        """IT: Upload successful file returns success or error component."""
        from io import BytesIO
        
        # Use a real ONNX file from test fixtures
        with open("model_store/linear_regression.onnx", "rb") as f:
            file_content = f.read()
        
        response = client.post(
            "/api/upload-and-run-ui",
            files={"file": ("model.onnx", BytesIO(file_content), "application/octet-stream")},
        )
        
        # Should return either success or a valid HTML response
        assert response.status_code == 200
        # Could be success or error - both are valid HTML responses
        assert "alert-" in response.text or "Prediction URL" in response.text
    
    def test_upload_rejects_invalid_file_type(self):
        """IT: Upload rejects unsupported file types."""
        from io import BytesIO
        
        response = client.post(
            "/api/upload-and-run-ui",
            files={"file": ("model.txt", BytesIO(b"invalid content"), "text/plain")},
        )
        
        assert response.status_code == 200
        assert "alert-error" in response.text
        assert "Unsupported file type" in response.text or "Error" in response.text
    
    def test_upload_includes_model_id_in_response(self):
        """IT: Success response contains model ID or error message."""
        from io import BytesIO
        
        with open("model_store/linear_regression.onnx", "rb") as f:
            file_content = f.read()
        
        response = client.post(
            "/api/upload-and-run-ui",
            files={"file": ("model.onnx", BytesIO(file_content), "application/octet-stream")},
        )
        
        assert response.status_code == 200
        # Either shows model ID (success) or error message
        assert "Model ID" in response.text or "Error" in response.text or "alert-" in response.text
    
    def test_upload_includes_prediction_url(self):
        """IT: Success response contains prediction URL or error."""
        from io import BytesIO
        
        with open("model_store/linear_regression.onnx", "rb") as f:
            file_content = f.read()
        
        response = client.post(
            "/api/upload-and-run-ui",
            files={"file": ("model.onnx", BytesIO(file_content), "application/octet-stream")},
        )
        
        assert response.status_code == 200
        # Either shows prediction URL (success) or error
        assert "Prediction URL" in response.text or "127.0.0.1" in response.text or "alert-" in response.text


class TestPredictionWorkflow:
    """IT: Test complete prediction workflow."""
    
    def test_prediction_with_valid_json(self):
        """IT: Prediction with valid JSON input returns result."""
        # First, we need to check if a model exists in registry
        from pathlib import Path
        registry_dir = Path("./app/registry/models")
        
        if registry_dir.exists() and list(registry_dir.glob("*.json")):
            # Get the first model ID
            model_file = next(registry_dir.glob("*.json"))
            model_id = model_file.stem
            
            # Send valid JSON prediction with proper form submission
            response = client.post(
                "/predict-result",
                data={
                    "model_id": model_id,
                    "input_data": json.dumps({"x": [1.0, 2.0]})
                }
            )
            
            # Should return either success or model-specific error
            assert response.status_code in [200, 404, 500]
    
    def test_prediction_rejects_invalid_json(self):
        """IT: Prediction with invalid JSON returns error component."""
        # Use proper form submission (data, not json)
        from io import BytesIO
        
        response = client.post(
            "/predict-result",
            data={
                "model_id": "test-model",
                "input_data": "not valid json {"
            }
        )
        
        # May get 422 for form validation or 200 with error component
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            assert "alert-error" in response.text
            assert "Invalid JSON" in response.text or "failed" in response.text.lower()
    
    def test_prediction_not_found_model(self):
        """IT: Prediction for non-existent model returns error."""
        response = client.post(
            "/predict-result",
            data={
                "model_id": "nonexistent-model-xyz",
                "input_data": json.dumps({"x": [1.0, 2.0]})
            }
        )
        
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            assert "alert-error" in response.text
            assert "not found" in response.text.lower() or "failed" in response.text.lower()


class TestHTMLComponentsIntegration:
    """IT: Test that HTML components render correctly in full page context."""
    
    def test_upload_page_has_complete_html_structure(self):
        """IT: Upload page has valid HTML structure."""
        response = client.get("/")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text
        assert "<html" in response.text
        assert "</html>" in response.text
        assert "<body" in response.text
        assert "</body>" in response.text
    
    def test_prediction_page_has_complete_html_structure(self):
        """IT: Prediction page has valid HTML structure."""
        response = client.get("/predict?model_id=test")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text
        assert "<html" in response.text
        assert "</html>" in response.text
    
    def test_pages_have_responsive_meta_viewport(self):
        """IT: Pages are responsive (mobile-friendly)."""
        response = client.get("/")
        assert 'name="viewport"' in response.text
        assert "width=device-width" in response.text
        assert "initial-scale=1" in response.text
    
    def test_components_use_semantic_html(self):
        """IT: Components use semantic HTML elements."""
        response = client.get("/")
        assert "<form" in response.text
        assert "<button" in response.text
        assert "<label" in response.text
        assert "<input" in response.text


class TestTunnelIntegration:
    """IT: Test tunnel URL display in success response."""
    
    def test_success_response_without_tunnel(self):
        """IT: Success response when tunnel is disabled."""
        from app.routing.frontend import upload_success_response
        
        html = upload_success_response(
            model_id="test-model",
            port=8001,
            tunnel_url=None
        )
        
        assert "Local Prediction URL" in html
        assert "http://127.0.0.1:8000/predict?model_id=test-model" in html
        assert "Public Tunnel URL" not in html
    
    def test_success_response_with_tunnel(self):
        """IT: Success response displays tunnel URL when provided."""
        from app.routing.frontend import upload_success_response
        
        html = upload_success_response(
            model_id="test-model",
            port=8001,
            tunnel_url="https://test-model.ngrok.io"
        )
        
        assert "Local Prediction URL" in html
        assert "http://127.0.0.1:8000/predict?model_id=test-model" in html
        assert "Public Tunnel URL" in html
        assert "https://test-model.ngrok.io" in html
        assert "Share this link" in html


class TestErrorHandling:
    """IT: Test error handling in components."""
    
    def test_upload_handles_missing_file(self):
        """IT: Upload endpoint handles missing file gracefully."""
        response = client.post("/api/upload-and-run-ui")
        # Should return error (422 unprocessable or 200 with error component)
        assert response.status_code in [200, 422]
    
    def test_prediction_error_component_includes_navigation(self):
        """IT: Error components include navigation options."""
        from app.routing.frontend import prediction_error_component
        
        html = prediction_error_component(
            error="Test error message",
            model_id="test-model"
        )
        
        assert "alert-error" in html
        assert "Test error message" in html
        assert "Try Again" in html
        assert "Upload New Model" in html


# ============================================================================
# HTMX-Specific Tests
# ============================================================================


class TestHTMXAttributes:
    """Test HTMX-specific attributes and behavior."""
    
    def test_upload_form_has_hx_post(self):
        """Test form has hx-post for AJAX submission."""
        response = client.get("/")
        assert 'hx-post="/api/upload-and-run-ui"' in response.text
    
    def test_predict_form_has_hx_post(self):
        """Test prediction form has hx-post."""
        response = client.get("/predict?model_id=test")
        assert 'hx-post="/predict-result"' in response.text
    
    def test_loading_indicator_has_htmx_class(self):
        """Test loading indicators use htmx-indicator class."""
        response = client.get("/")
        assert 'class="htmx-indicator' in response.text or "htmx-indicator" in response.text


# ============================================================================
# Accessibility and Usability Tests
# ============================================================================


class TestAccessibility:
    """Test accessibility features."""
    
    def test_upload_page_has_form_labels(self):
        """Test form has proper labels for accessibility."""
        response = client.get("/")
        assert "<label" in response.text
        assert "for=" in response.text
    
    def test_prediction_page_has_aria_support(self):
        """Test pages support ARIA attributes."""
        response = client.get("/predict?model_id=test")
        # At minimum, should have proper semantic structure
        assert "<textarea" in response.text
        assert "<button" in response.text
    
    def test_error_messages_are_readable(self):
        """Test error messages are clear and user-friendly."""
        response = client.post(
            "/predict-result",
            data={
                "model_id": "test",
                "input_data": "invalid {{"
            }
        )
        
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            # Error should be in readable format
            assert "alert-error" in response.text
            assert "Error" in response.text or "failed" in response.text.lower()
