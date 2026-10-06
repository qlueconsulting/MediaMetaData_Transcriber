"""Unit tests for the common persistent storage structure at /srv/storage/jobs/{job_id}/."""

import pytest
from pathlib import Path
from app.services.storage import StorageService


class TestStorageService:
    """Tests for job directory structure and artifact persistence."""

    def test_save_and_get_meta(self, tmp_path):
        """meta.json must be saved and retrievable."""
        storage = StorageService(base_dir=tmp_path)
        job_id = "test-job-001"
        data = {
            "title": "Test Title",
            "creator": "Test Creator",
            "duration_seconds": 300,
            "platform": "youtube",
        }

        meta_path = storage.save_meta(job_id, data)
        assert meta_path.is_file()
        assert meta_path.name == "meta.json"
        assert meta_path.parent.name == job_id

        loaded = storage.get_meta(job_id)
        assert loaded is not None
        assert loaded["title"] == "Test Title"
        assert loaded["duration_seconds"] == 300

    def test_audio_file_checks(self, tmp_path):
        """audio.mp3 path and size retrieval."""
        storage = StorageService(base_dir=tmp_path)
        job_id = "test-job-002"

        # Initially no audio
        assert storage.has_audio(job_id) is False
        assert storage.get_audio_size(job_id) is None

        # Create dummy audio file
        audio_path = storage.get_audio_path(job_id)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"\x00" * 1024)

        assert storage.has_audio(job_id) is True
        assert storage.get_audio_size(job_id) == 1024

    def test_save_and_get_transcript(self, tmp_path):
        """transcript.json must be saved and retrievable."""
        storage = StorageService(base_dir=tmp_path)
        job_id = "test-job-003"
        data = {
            "job_id": job_id,
            "text": "Hello world, this is a test transcript.",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world,"},
                {"id": 1, "start": 2.5, "end": 5.0, "text": " this is a test transcript."},
            ],
        }

        transcript_path = storage.save_transcript(job_id, data)
        assert transcript_path.is_file()
        assert transcript_path.name == "transcript.json"

        loaded = storage.get_transcript(job_id)
        assert loaded is not None
        assert loaded["language"] == "en"
        assert len(loaded["segments"]) == 2

    def test_job_summary_and_deletion(self, tmp_path):
        """Check summary of all 3 artifacts and job cleanup."""
        storage = StorageService(base_dir=tmp_path)
        job_id = "test-job-004"

        # Save meta, audio, transcript
        storage.save_meta(job_id, {"title": "Full Test"})
        storage.get_audio_path(job_id).write_bytes(b"dummy audio")
        storage.save_transcript(job_id, {"text": "dummy text"})

        summary = storage.get_job_summary(job_id)
        assert summary["exists"] is True
        assert summary["meta_exists"] is True
        assert summary["audio_exists"] is True
        assert summary["transcript_exists"] is True

        jobs = storage.list_jobs()
        assert any(j["job_id"] == job_id for j in jobs)

        # Delete job
        assert storage.delete_job(job_id) is True
        assert storage.get_job_summary(job_id)["exists"] is False

    def test_find_job_by_url_and_caching(self, tmp_path):
        """find_job_by_url must find cached jobs by exact and normalized URLs with optional file requirements."""
        from app.services.storage import normalize_url

        # Test URL normalization
        assert normalize_url("https://youtu.be/dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert normalize_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        storage = StorageService(base_dir=tmp_path)
        job_id = "job-cache-001"
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # Not found initially
        assert storage.find_job_by_url(url) is None

        # Save metadata
        storage.save_meta(job_id, {"url": url, "title": "Never Gonna Give You Up"})

        # Found by full URL and by youtu.be short URL
        assert storage.find_job_by_url(url) == job_id
        assert storage.find_job_by_url("https://youtu.be/dQw4w9WgXcQ") == job_id

        # require_audio fails because audio.mp3 doesn't exist yet
        assert storage.find_job_by_url(url, require_audio=True) is None

        # Add audio.mp3
        storage.get_audio_path(job_id).write_bytes(b"\x00" * 100)
        assert storage.find_job_by_url(url, require_audio=True) == job_id

        # require_transcript fails until transcript.json exists
        assert storage.find_job_by_url(url, require_transcript=True) is None
        storage.save_transcript(job_id, {"text": "Never gonna let you down"})
        assert storage.find_job_by_url(url, require_transcript=True) == job_id

