"""Service for managing the persistent storage structure at /srv/storage/jobs/{job_id}/."""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from app.config import settings
from app.utils.logger import log


class StorageService:
    """Manages files in /srv/storage/jobs/{job_id}/:

    - meta.json
    - audio.mp3
    - transcript.json
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or settings.STORAGE_DIR)
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        """Create the root jobs directory if it does not exist."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning(f"Could not create storage root {self.base_dir}: {e}")

    def get_job_dir(self, job_id: str, create: bool = True) -> Path:
        """Return the directory path for a specific job."""
        job_dir = self.base_dir / str(job_id)
        if create:
            job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_meta_path(self, job_id: str) -> Path:
        """Return path to meta.json for a job."""
        return self.get_job_dir(job_id, create=False) / "meta.json"

    def get_audio_path(self, job_id: str) -> Path:
        """Return path to audio.mp3 for a job."""
        return self.get_job_dir(job_id, create=False) / "audio.mp3"

    def get_transcript_path(self, job_id: str) -> Path:
        """Return path to transcript.json for a job."""
        return self.get_job_dir(job_id, create=False) / "transcript.json"

    def save_meta(self, job_id: str, data: Dict[str, Any]) -> Path:
        """Save video metadata to /srv/storage/jobs/{job_id}/meta.json."""
        self.get_job_dir(job_id, create=True)
        meta_path = self.get_meta_path(job_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"Saved meta.json for job {job_id} at {meta_path}")
        return meta_path

    def get_meta(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Read and parse meta.json for a job."""
        meta_path = self.get_meta_path(job_id)
        if not meta_path.is_file():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_transcript(self, job_id: str, data: Dict[str, Any]) -> Path:
        """Save full transcript and segment timestamps to /srv/storage/jobs/{job_id}/transcript.json."""
        self.get_job_dir(job_id, create=True)
        transcript_path = self.get_transcript_path(job_id)
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"Saved transcript.json for job {job_id} at {transcript_path}")
        return transcript_path

    def get_transcript(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Read and parse transcript.json for a job."""
        transcript_path = self.get_transcript_path(job_id)
        if not transcript_path.is_file():
            return None
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_audio(self, job_id: str) -> bool:
        """Check if audio.mp3 exists and is non-empty."""
        audio_path = self.get_audio_path(job_id)
        return audio_path.is_file() and audio_path.stat().st_size > 0

    def get_audio_size(self, job_id: str) -> Optional[int]:
        """Return size in bytes of audio.mp3."""
        audio_path = self.get_audio_path(job_id)
        if audio_path.is_file():
            return audio_path.stat().st_size
        return None

    def get_job_summary(self, job_id: str) -> Dict[str, Any]:
        """Check status of all three required files for a job."""
        job_dir = self.get_job_dir(job_id, create=False)
        meta_exists = self.get_meta_path(job_id).is_file()
        audio_exists = self.get_audio_path(job_id).is_file()
        transcript_exists = self.get_transcript_path(job_id).is_file()

        return {
            "job_id": job_id,
            "exists": job_dir.is_dir(),
            "job_dir": str(job_dir),
            "meta_exists": meta_exists,
            "audio_exists": audio_exists,
            "transcript_exists": transcript_exists,
            "audio_size_bytes": self.get_audio_size(job_id) if audio_exists else None,
        }

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent jobs in the storage directory."""
        if not self.base_dir.is_dir():
            return []

        jobs = []
        for child in sorted(self.base_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if child.is_dir():
                jobs.append(self.get_job_summary(child.name))
                if len(jobs) >= limit:
                    break
        return jobs

    def delete_job(self, job_id: str) -> bool:
        """Delete all artifacts for a job."""
        job_dir = self.get_job_dir(job_id, create=False)
        if job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
            log.info(f"Deleted job directory for {job_id}")
            return True
        return False


# Singleton instance
storage_service = StorageService()
