"""Media metadata extraction API endpoint."""

from fastapi import APIRouter
from app.models.metadata import MediaMetadataRequest, MediaMetadataResponse
from app.services.media import media_service

router = APIRouter(tags=["Metadata"])


@router.post("/metadata", response_model=MediaMetadataResponse)
def get_media_metadata(request: MediaMetadataRequest) -> MediaMetadataResponse:
    """Extract metadata for a target media URL.

    - Resolves title, creator, duration, platform, timestamps, thumbnail without downloading the stream.
    - Enforces the Early 20-Minute Decision Tree: Videos > 1,200s (20 mins) are flagged immediately.
    """
    return media_service.extract_metadata(
        url=str(request.url).strip(),
        bypass_cache=request.bypass_cache,
    )
