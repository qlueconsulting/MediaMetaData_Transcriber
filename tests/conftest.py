"""Pytest configuration and shared fixtures."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def override_storage_dirs(monkeypatch, tmp_path):
    """Ensure all tests run using isolated temporary storage directories."""
    test_jobs_dir = tmp_path / "jobs"
    test_models_dir = tmp_path / "models"
    test_jobs_dir.mkdir(parents=True, exist_ok=True)
    test_models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "STORAGE_DIR", test_jobs_dir)
    monkeypatch.setattr(settings, "MODEL_DIR", test_models_dir)

    from app.services.storage import storage_service
    monkeypatch.setattr(storage_service, "base_dir", test_jobs_dir)

    yield test_jobs_dir


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_meta_dict():
    """Sample valid metadata dictionary for a 10-minute video."""
    return {
        "title": "Introduction to AI Audio Processing",
        "uploader": "TechCreator",
        "creator": "TechCreator",
        "duration": 600,  # 10 minutes (well under 1,200s limit)
        "extractor_key": "Youtube",
        "extractor": "youtube",
        "upload_date": "20260101",
        "view_count": 50000,
        "thumbnail": "https://example.com/thumb.jpg",
        "description": "A tutorial on AI speech processing.",
    }


@pytest.fixture
def sample_oversized_meta_dict():
    """Sample metadata dictionary for a 25-minute video (> 1,200 seconds)."""
    return {
        "title": "Long 25-Minute Podcast Episode",
        "uploader": "PodcastStudio",
        "creator": "PodcastStudio",
        "duration": 1500,  # 25 minutes (exceeds 1,200s limit)
        "extractor_key": "Youtube",
        "extractor": "youtube",
        "upload_date": "20260101",
        "view_count": 12000,
        "thumbnail": "https://example.com/podcast.jpg",
        "description": "A long podcast episode.",
    }
