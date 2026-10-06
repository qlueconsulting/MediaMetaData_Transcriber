"""Pydantic models for media metadata extraction."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class MediaMetadataRequest(BaseModel):
    """Request payload to extract media metadata."""
    url: str = Field(
        ...,
        description="Public URL of the media (YouTube, Vimeo, Twitter, direct stream, etc.)",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    bypass_cache: bool = Field(
        default=False,
        description="If True, bypasses any cached metadata and forces a fresh query to the provider",
    )
    max_duration_minutes: Optional[int] = Field(
        default=None,
        description="Optional override for maximum video duration in minutes. 0 = unlimited. If omitted, defaults to 20.",
    )


class MediaMetadataResponse(BaseModel):
    """Extracted metadata adhering to the microservice specifications."""
    job_id: Optional[str] = Field(
        default=None,
        description="Associated job ID if created as part of a job pipeline",
    )
    url: str = Field(
        ...,
        description="Original media URL",
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of the media",
    )
    creator: Optional[str] = Field(
        default=None,
        description="Primary creator or channel name",
    )
    uploader: Optional[str] = Field(
        default=None,
        description="Uploader name / channel identifier",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of the media in seconds",
    )
    duration_formatted: Optional[str] = Field(
        default=None,
        description="Human-readable duration (MM:SS or HH:MM:SS)",
    )
    platform: Optional[str] = Field(
        default=None,
        description="Detected media platform (e.g. youtube, soundcloud, etc.)",
    )
    extractor: Optional[str] = Field(
        default=None,
        description="yt-dlp extractor key used",
    )
    upload_date: Optional[str] = Field(
        default=None,
        description="Upload date in YYYYMMDD format",
    )
    view_count: Optional[int] = Field(
        default=None,
        description="Total view count if available",
    )
    thumbnail: Optional[str] = Field(
        default=None,
        description="URL to the highest quality video thumbnail",
    )
    description: Optional[str] = Field(
        default=None,
        description="Summary description of the video",
    )
    # Operational Guardrails Flags
    exceeds_duration_limit: bool = Field(
        ...,
        description="Duration decision flag. True if duration > max_duration_seconds (when max_duration_seconds > 0)",
    )
    max_duration_seconds: Optional[int] = Field(
        default=1200,
        description="Max permitted duration threshold in seconds (0 = unlimited)",
    )
    allowed_for_transcription: bool = Field(
        ...,
        description="Whether this media is permitted for audio extraction and transcription",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Warning message if media exceeds guardrail threshold",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of metadata extraction",
    )
    cached: bool = Field(
        default=False,
        description="True if metadata was served from the local persistent storage cache",
    )
    cached_job_id: Optional[str] = Field(
        default=None,
        description="Job ID where cached metadata was retrieved from",
    )
    raw_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtered additional metadata key-values",
    )
