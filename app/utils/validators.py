"""Validation functions enforcing strict operational guardrails."""

import os
from pathlib import Path
from typing import Optional, Union
from fastapi import HTTPException, status
from app.config import settings
from app.utils.logger import log


class GuardrailViolationError(HTTPException):
    """Exception raised when operational guardrails are violated."""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


def format_duration(seconds: Optional[Union[int, float]]) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    if seconds is None:
        return "00:00"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def check_duration_limit(duration_seconds: Optional[Union[int, float]], max_seconds: Optional[int] = None) -> bool:
    """Check if the video duration exceeds the maximum allowed duration.

    Returns True if duration > max_seconds (when max_seconds > 0), False otherwise.
    If max_seconds <= 0, returns False (unlimited).
    If max_seconds is None, defaults to settings.MAX_DURATION_SECONDS.
    """
    if duration_seconds is None:
        return False
    if max_seconds is not None:
        if max_seconds <= 0:
            return False
        threshold = max_seconds
    else:
        threshold = settings.MAX_DURATION_SECONDS
    return float(duration_seconds) > float(threshold)


def validate_video_duration(
    duration_seconds: Optional[Union[int, float]],
    max_seconds: Optional[int] = None,
    context: str = "Video processing",
) -> None:
    """Enforce the Video Duration Guardrail.

    Any attempt to trigger audio extraction or transcription on a video exceeding max_seconds
    (default 1,200 seconds / 20 minutes) MUST be rejected early with an HTTP 400 Bad Request error.
    If max_seconds <= 0, the limit is bypassed (unlimited).
    """
    if duration_seconds is None:
        return

    if max_seconds is not None:
        if max_seconds <= 0:
            return
        threshold = max_seconds
    else:
        threshold = settings.MAX_DURATION_SECONDS

    if float(duration_seconds) > float(threshold):
        formatted_duration = format_duration(duration_seconds)
        max_minutes = threshold / 60.0
        error_msg = (
            f"{context} rejected: Video duration ({float(duration_seconds):.1f}s / {formatted_duration}) "
            f"exceeds the strict operational limit of {threshold} seconds ({int(max_minutes)} minutes)."
        )
        log.warning(f"Guardrail triggered: {error_msg}")
        raise GuardrailViolationError(detail=error_msg)


def validate_audio_file_size(
    file_path: Union[str, Path],
    max_bytes: Optional[int] = None,
) -> int:
    """Enforce the Audio File Size Guard (<= 25 MB).

    The service must enforce a strict maximum file size of 25 MB (26,214,400 bytes).
    If an audio file exceeds 25 MB, fail gracefully with an HTTP 400 error.
    """
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found at {path}",
        )

    file_size = path.stat().st_size
    threshold = max_bytes or settings.MAX_AUDIO_SIZE_BYTES
    max_mb = threshold / (1024 * 1024)
    file_size_mb = file_size / (1024 * 1024)

    if file_size > threshold:
        error_msg = (
            f"Audio processing rejected: Audio file size ({file_size_mb:.2f} MB / {file_size:,} bytes) "
            f"exceeds the strict operational limit of {max_mb:.2f} MB ({threshold:,} bytes)."
        )
        log.warning(f"Guardrail triggered: {error_msg}")
        # Clean up oversized file if needed
        try:
            if path.exists():
                path.unlink(missing_ok=True)
                log.info(f"Purged oversized audio artifact: {path}")
        except Exception as e:
            log.error(f"Failed to delete oversized file {path}: {e}")

        raise GuardrailViolationError(detail=error_msg)

    return file_size
