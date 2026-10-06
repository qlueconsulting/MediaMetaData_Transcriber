"""Export API endpoints."""

from app.api.endpoints.health import router as health_router
from app.api.endpoints.metadata import router as metadata_router
from app.api.endpoints.transcribe import router as transcribe_router
from app.api.endpoints.jobs import router as jobs_router

__all__ = ["health_router", "metadata_router", "transcribe_router", "jobs_router"]
