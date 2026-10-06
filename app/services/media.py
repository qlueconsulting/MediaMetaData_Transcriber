"""Media extraction service using yt-dlp and ffmpeg with operational guardrails."""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from fastapi import HTTPException, status
import yt_dlp

from app.config import settings
from app.utils.logger import log
from app.utils.validators import (
    validate_video_duration,
    validate_audio_file_size,
    check_duration_limit,
    format_duration,
)
from app.models.metadata import MediaMetadataResponse
from app.services.storage import storage_service


class MediaService:
    """Handles metadata resolution and audio extraction via yt-dlp and ffmpeg."""

    def __init__(self):
        pass

    def _get_base_ytdlp_opts(self) -> Dict[str, Any]:
        """Construct common yt-dlp options configured for residential IP operation."""
        clients = [c.strip() for c in settings.YTDLP_PLAYER_CLIENTS.split(",") if c.strip()]
        opts: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": settings.YTDLP_SOCKET_TIMEOUT,
            "retries": settings.YTDLP_RETRIES,
            "noplaylist": True,
            "extractor_args": {
                "youtube": {
                    "player_client": clients,
                }
            },
        }
        if settings.YTDLP_PROXY:
            opts["proxy"] = settings.YTDLP_PROXY
        return opts

    def extract_metadata(
        self,
        url: str,
        job_id: Optional[str] = None,
        bypass_cache: bool = False,
        max_duration_minutes: Optional[int] = None,
    ) -> MediaMetadataResponse:
        """Extract media metadata without downloading the full stream.

        Enforces video duration limit (default 1,200s / 20 mins, overridable via max_duration_minutes).
        Supports cache lookup unless bypass_cache=True.
        """
        if max_duration_minutes is not None:
            max_seconds = max_duration_minutes * 60 if max_duration_minutes > 0 else 0
        else:
            max_seconds = settings.MAX_DURATION_SECONDS

        if not bypass_cache:
            cached_job_id = storage_service.find_job_by_url(url, require_audio=False)
            if cached_job_id:
                meta_dict = storage_service.get_meta(cached_job_id)
                if meta_dict:
                    log.info(f"Serving cached metadata for {url} from job {cached_job_id}")
                    cached_meta = MediaMetadataResponse(**meta_dict)
                    cached_meta.cached = True
                    cached_meta.cached_job_id = cached_job_id
                    if job_id:
                        cached_meta.job_id = job_id
                    if max_duration_minutes is not None:
                        cached_meta.max_duration_seconds = max_seconds
                        cached_meta.exceeds_duration_limit = check_duration_limit(cached_meta.duration_seconds, max_seconds)
                        cached_meta.allowed_for_transcription = not cached_meta.exceeds_duration_limit
                        if cached_meta.exceeds_duration_limit:
                            limit_desc = f"{max_seconds} seconds ({max_duration_minutes} minutes)" if max_seconds > 0 else "limit"
                            cached_meta.warning = (
                                f"Video duration ({cached_meta.duration_seconds:.1f}s / {cached_meta.duration_formatted}) "
                                f"exceeds the strict limit of {limit_desc}. "
                                f"Audio extraction and transcription are rejected."
                            )
                        else:
                            cached_meta.warning = None
                    return cached_meta

        log.info(f"Extracting metadata for URL: {url}")
        opts = self._get_base_ytdlp_opts()
        opts.update({
            "skip_download": True,
            "extract_flat": False,
        })

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            log.error(f"yt-dlp extraction failed for {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to resolve media metadata from URL: {str(e)}",
            )
        except Exception as e:
            log.error(f"Unexpected error resolving metadata for {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error processing media: {str(e)}",
            )

        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Could not extract metadata from the provided URL",
            )

        # Handle entries if playlist was returned despite noplaylist
        if "entries" in info and info["entries"]:
            info = info["entries"][0]

        duration = info.get("duration")
        duration_sec = float(duration) if duration is not None else None
        exceeds_limit = check_duration_limit(duration_sec, max_seconds)
        formatted_duration = format_duration(duration_sec)

        warning_msg = None
        if exceeds_limit:
            limit_desc = f"{max_seconds} seconds ({max_seconds // 60} minutes)" if max_seconds > 0 else "limit"
            warning_msg = (
                f"Video duration ({duration_sec:.1f}s / {formatted_duration}) exceeds the strict limit "
                f"of {limit_desc}. Audio extraction and transcription are rejected."
            )
            log.warning(f"Duration Flag triggered: {warning_msg}")

        metadata = MediaMetadataResponse(
            job_id=job_id,
            url=url,
            title=info.get("title"),
            creator=info.get("creator") or info.get("uploader") or info.get("channel"),
            uploader=info.get("uploader") or info.get("channel"),
            duration_seconds=duration_sec,
            duration_formatted=formatted_duration,
            platform=info.get("extractor_key") or info.get("extractor"),
            extractor=info.get("extractor"),
            upload_date=info.get("upload_date"),
            view_count=info.get("view_count"),
            thumbnail=info.get("thumbnail"),
            description=info.get("description"),
            exceeds_duration_limit=exceeds_limit,
            max_duration_seconds=max_seconds,
            allowed_for_transcription=not exceeds_limit,
            warning=warning_msg,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return metadata

    def download_and_extract_audio(
        self,
        url: str,
        job_id: str,
        pre_extracted_meta: Optional[MediaMetadataResponse] = None,
        bypass_cache: bool = False,
        max_duration_minutes: Optional[int] = None,
    ) -> Tuple[Path, MediaMetadataResponse]:
        """Download and extract audio converted to 16kHz mono MP3 (-ac 1 -ar 16000 -b:a 64k).

        Enforces:
        1. Video duration limit (default 20 mins / 1,200s, overridable via max_duration_minutes).
        2. Audio file size <= 25MB check with HTTP 400.
        3. Saves meta.json and audio.mp3 into /srv/storage/jobs/{job_id}/.
        Supports cache lookup unless bypass_cache=True.
        """
        job_dir = storage_service.get_job_dir(job_id, create=True)
        target_audio_path = job_dir / "audio.mp3"

        if max_duration_minutes is not None:
            max_seconds = max_duration_minutes * 60 if max_duration_minutes > 0 else 0
        else:
            max_seconds = settings.MAX_DURATION_SECONDS

        if not bypass_cache:
            cached_job_id = storage_service.find_job_by_url(url, require_audio=True)
            if cached_job_id:
                cached_audio = storage_service.get_audio_path(cached_job_id)
                cached_meta_dict = storage_service.get_meta(cached_job_id)
                if cached_audio.is_file() and cached_meta_dict:
                    log.info(f"Serving cached audio for {url} from job {cached_job_id}")
                    if cached_job_id != job_id:
                        shutil.copy2(cached_audio, target_audio_path)
                        storage_service.save_meta(job_id, {**cached_meta_dict, "job_id": job_id})
                    else:
                        target_audio_path = cached_audio
                    cached_meta = MediaMetadataResponse(**cached_meta_dict)
                    cached_meta.job_id = job_id
                    cached_meta.cached = True
                    cached_meta.cached_job_id = cached_job_id
                    if max_duration_minutes is not None:
                        cached_meta.max_duration_seconds = max_seconds
                        cached_meta.exceeds_duration_limit = check_duration_limit(cached_meta.duration_seconds, max_seconds)
                        cached_meta.allowed_for_transcription = not cached_meta.exceeds_duration_limit
                    return target_audio_path, cached_meta

        # 1. Video Duration Guardrail Check
        if pre_extracted_meta is not None:
            metadata = pre_extracted_meta
            metadata.job_id = job_id
        else:
            metadata = self.extract_metadata(
                url,
                job_id=job_id,
                bypass_cache=bypass_cache,
                max_duration_minutes=max_duration_minutes,
            )

        # Reject if > max_seconds
        validate_video_duration(
            duration_seconds=metadata.duration_seconds,
            max_seconds=max_seconds,
            context="Audio extraction",
        )

        # Save meta.json to /srv/storage/jobs/{job_id}/meta.json
        storage_service.save_meta(job_id, metadata.model_dump())

        # 2. Configure yt-dlp for 16kHz Mono MP3 @ 64kbps conversion
        output_template = str(job_dir / "audio.%(ext)s")
        target_audio_path = job_dir / "audio.mp3"

        # Remove any pre-existing temporary audio file
        if target_audio_path.exists():
            target_audio_path.unlink()

        opts = self._get_base_ytdlp_opts()
        opts.update({
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }
            ],
            "postprocessor_args": [
                "-ac", str(settings.AUDIO_CHANNELS),
                "-ar", str(settings.AUDIO_SAMPLE_RATE),
                "-b:a", settings.AUDIO_BITRATE,
            ],
            "keepvideo": False,
        })

        log.info(
            f"Downloading and converting audio for job {job_id} "
            f"(-ac {settings.AUDIO_CHANNELS} -ar {settings.AUDIO_SAMPLE_RATE} -b:a {settings.AUDIO_BITRATE})"
        )

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as e:
            log.error(f"Download/conversion failed for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio extraction failed: {str(e)}",
            )
        except Exception as e:
            log.error(f"Unexpected error during audio extraction for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio extraction processing error: {str(e)}",
            )

        if not target_audio_path.is_file():
            # Check if yt-dlp saved under a slightly different name
            mp3_files = list(job_dir.glob("*.mp3"))
            if mp3_files:
                mp3_files[0].rename(target_audio_path)
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Audio extraction completed but audio.mp3 was not generated",
                )

        # 3. Audio File Size Guard (<= 25 MB)
        file_size = validate_audio_file_size(
            target_audio_path,
            max_bytes=settings.MAX_AUDIO_SIZE_BYTES,
        )

        log.info(
            f"Audio extracted successfully for job {job_id}: "
            f"{target_audio_path} ({file_size / (1024 * 1024):.2f} MB)"
        )

        return target_audio_path, metadata


# Singleton instance
media_service = MediaService()
