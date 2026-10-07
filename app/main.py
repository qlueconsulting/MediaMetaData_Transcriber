"""MediaMetaData_Transcriber FastAPI Application Entrypoint."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils.logger import log
from app.api.router import api_router
from app.services.whisper_local import local_whisper_service
from app.services.groq_client import groq_whisper_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    log.info("Starting MediaMetaData_Transcriber Microservice...")

    # Ensure storage paths exist
    settings.ensure_directories()
    log.info(f"Storage directory: {settings.STORAGE_DIR}")
    log.info(f"Model cache directory: {settings.MODEL_DIR}")

    # Log hardware and engine diagnostics
    cuda_available, cuda_info = local_whisper_service.is_cuda_available()
    log.info(f"CUDA status: {cuda_info}")
    log.info(f"Local Whisper config: model={settings.WHISPER_MODEL}, device={settings.WHISPER_DEVICE}, compute_type={settings.WHISPER_COMPUTE_TYPE}")

    if groq_whisper_service.is_configured:
        log.info(f"Groq Whisper Cloud Fallback: Enabled (Model: {settings.GROQ_MODEL})")
    else:
        log.warning("Groq Whisper Cloud Fallback: Unconfigured (GROQ_API_KEY is not set)")

    log.info(f"Operational Guardrails: MAX_DURATION={settings.MAX_DURATION_SECONDS}s (20m), MAX_AUDIO_SIZE={settings.MAX_AUDIO_SIZE_BYTES} bytes (25MB)")

    # Warm up Whisper model in a background thread so container starts immediately
    if "PYTEST_CURRENT_TEST" not in os.environ and not os.environ.get("TESTING"):
        import threading

        def _warmup():
            try:
                log.info(f"Asynchronously warming up Whisper model '{settings.WHISPER_MODEL}'...")
                local_whisper_service.load_model()
                log.info(f"Whisper model '{settings.WHISPER_MODEL}' warm and ready in memory.")
            except Exception as e:
                log.warning(f"Background model warmup deferred: {e}")

        threading.Thread(target=_warmup, daemon=True).start()

    yield

    log.info("Shutting down MediaMetaData_Transcriber Microservice...")


app = FastAPI(
    title="MediaMetaData_Transcriber",
    description=(
        "High-performance self-hosted GPU-accelerated video metadata extraction, "
        "audio processing, and Whisper transcription microservice."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware (Configured for reverse proxy / OPNsense integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_api_key_middleware(request: Request, call_next):
    """Enforce optional API key authentication if settings.API_KEY is configured."""
    if settings.API_KEY and settings.API_KEY.strip():
        public_paths = {
            "/",
            "/ui",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/health",
            "/health",
        }
        path = request.url.path.rstrip("/")
        normalized_path = path if path else "/"

        if normalized_path not in public_paths and not request.url.path.startswith("/static"):
            provided = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
            if not provided:
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    provided = auth[7:].strip()

            if not provided or provided.strip() != settings.API_KEY.strip():
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: Invalid or missing API key. Provide via 'X-API-Key' header."},
                )

    return await call_next(request)


# Register main API router under both /api/v1 and /api for compatibility
app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router, prefix="/api")


from pathlib import Path
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

STATIC_INDEX_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", tags=["Root & Testing Apparatus"])
def root(request: Request):
    """Serve embedded Testing Apparatus web UI for browsers, or return JSON status for API clients."""
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header and STATIC_INDEX_PATH.is_file():
        return FileResponse(STATIC_INDEX_PATH, media_type="text/html")

    return JSONResponse({
        "service": "MediaMetaData_Transcriber",
        "status": "online",
        "version": "1.0.0",
        "testing_ui": "/ui",
        "docs": "/docs",
        "health": "/api/v1/health",
        "storage": str(settings.STORAGE_DIR),
        "guardrails": {
            "max_video_duration_seconds": settings.MAX_DURATION_SECONDS,
            "max_audio_size_bytes": settings.MAX_AUDIO_SIZE_BYTES,
        },
    })


@app.get("/ui", tags=["Root & Testing Apparatus"], response_class=HTMLResponse)
@app.get("/test", tags=["Root & Testing Apparatus"], response_class=HTMLResponse)
def testing_ui():
    """Direct route for Testing Apparatus web interface."""
    if STATIC_INDEX_PATH.is_file():
        return FileResponse(STATIC_INDEX_PATH, media_type="text/html")
    return HTMLResponse("<h1>Testing Apparatus UI not found</h1>", status_code=404)


@app.get("/health", tags=["Health & Diagnostics"])
def root_health():
    """Convenience alias for /api/v1/health."""
    from app.api.endpoints.health import health_check
    return health_check()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
