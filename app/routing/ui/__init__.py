"""Frontend UI package."""

from app.routing.ui.router import router
from app.routing.ui.templates import prediction_error_component, upload_success_response

__all__ = ["router", "upload_success_response", "prediction_error_component"]
