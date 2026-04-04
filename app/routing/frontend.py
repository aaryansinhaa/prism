"""Compatibility wrapper for frontend routes.

Actual implementation lives under `app.routing.ui` to keep files small and clean.
"""

from app.routing.ui import prediction_error_component, router, upload_success_response

__all__ = ["router", "upload_success_response", "prediction_error_component"]
