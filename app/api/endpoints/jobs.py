"""Job management and artifact retrieval endpoints."""

from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

from app.services.storage import storage_service
from app.utils.logger import log

router = APIRouter(tags=["Jobs & Storage"])


def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


@router.get("/jobs")
def list_jobs(limit: int = Query(default=50, ge=1, le=500)):
    """List recent jobs stored in /srv/storage/jobs/."""
    return storage_service.list_jobs(limit=limit)


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieve file existence and status summary for a job."""
    summary = storage_service.get_job_summary(job_id)
    if not summary["exists"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found in persistent storage",
        )
    return summary


@router.get("/jobs/{job_id}/meta")
def get_job_metadata(job_id: str):
    """Retrieve meta.json for a job."""
    meta = storage_service.get_meta(job_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"meta.json for job {job_id} not found",
        )
    return meta


@router.get("/jobs/{job_id}/audio")
def get_job_audio(job_id: str):
    """Stream or download audio.mp3 for a job."""
    audio_path = storage_service.get_audio_path(job_id)
    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"audio.mp3 for job {job_id} not found",
        )
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=f"{job_id}.mp3",
    )


@router.get("/jobs/{job_id}/transcript")
def get_job_transcript(
    job_id: str,
    format: str = Query(
        default="json",
        description="Response format: 'json', 'text', 'srt', or 'vtt'",
    ),
):
    """Retrieve transcript.json in JSON, plain text, SubRip (SRT), or WebVTT format."""
    transcript = storage_service.get_transcript(job_id)
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"transcript.json for job {job_id} not found",
        )

    fmt = format.lower().strip()

    if fmt == "json":
        return transcript

    if fmt in ["text", "txt"]:
        return PlainTextResponse(
            content=transcript.get("text", ""),
            media_type="text/plain; charset=utf-8",
        )

    segments = transcript.get("segments", [])

    if fmt == "srt":
        srt_lines = []
        for idx, seg in enumerate(segments, 1):
            start_str = seconds_to_srt_timestamp(seg.get("start", 0.0))
            end_str = seconds_to_srt_timestamp(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            srt_lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        return PlainTextResponse(
            content="\n".join(srt_lines),
            media_type="application/x-subrip; charset=utf-8",
        )

    if fmt == "vtt":
        vtt_lines = ["WEBVTT\n"]
        for seg in segments:
            start_str = seconds_to_vtt_timestamp(seg.get("start", 0.0))
            end_str = seconds_to_vtt_timestamp(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            vtt_lines.append(f"{start_str} --> {end_str}\n{text}\n")
        return PlainTextResponse(
            content="\n".join(vtt_lines),
            media_type="text/vtt; charset=utf-8",
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported format '{format}'. Allowed formats: json, text, srt, vtt",
    )


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete all files and directory for a job."""
    deleted = storage_service.delete_job(job_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return {"status": "deleted", "job_id": job_id}
