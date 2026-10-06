"""Central API Router for MediaMetaData_Transcriber."""

from fastapi import APIRouter
from app.api.endpoints.health import router as health_router
from app.api.endpoints.metadata import router as metadata_router
from app.api.endpoints.transcribe import router as transcribe_router
from app.api.endpoints.jobs import router as jobs_router

api_router = APIRouter()

# Group all v1 routes
api_router.include_router(health_router, prefix="", tags=["Health & Diagnostics"])
api_router.include_router(metadata_router, prefix="", tags=["Metadata"])
api_router.include_router(transcribe_router, prefix="", tags=["Transcription"])
api_router.include_router(jobs_router, prefix="", tags=["Jobs & Storage"])
