"""Unit tests for the strict operational guardrails:

1. Early 20-Minute Decision Tree (> 1,200 seconds rejection)
2. Audio File Size Guard (<= 25 MB rejection)
"""

import pytest
from pathlib import Path
from fastapi import HTTPException
from app.config import settings
from app.utils.validators import (
    validate_video_duration,
    validate_audio_file_size,
    check_duration_limit,
    format_duration,
    GuardrailViolationError,
)


class TestDurationGuardrail:
    """Tests for the Early 20-Minute Decision Tree."""

    def test_duration_under_20_minutes_passes(self):
        """Videos under 1,200 seconds should pass validation."""
        # 10 minutes = 600s
        assert check_duration_limit(600, max_seconds=1200) is False
        # Does not raise
        validate_video_duration(600, max_seconds=1200)

    def test_duration_exactly_20_minutes_passes(self):
        """Videos exactly 1,200 seconds (20 minutes) should pass validation."""
        assert check_duration_limit(1200, max_seconds=1200) is False
        validate_video_duration(1200, max_seconds=1200)

    def test_duration_over_20_minutes_flags_and_raises_http_400(self):
        """Videos > 1,200 seconds must trigger check_duration_limit and raise HTTP 400."""
        # 1,201 seconds = 20 minutes 1 second
        assert check_duration_limit(1201, max_seconds=1200) is True

        with pytest.raises(GuardrailViolationError) as exc_info:
            validate_video_duration(1201, max_seconds=1200)

        assert exc_info.value.status_code == 400
        assert "exceeds the strict operational limit of 1200 seconds (20 minutes)" in exc_info.value.detail

    def test_duration_override_custom_limit_passes(self):
        """Videos over default 20m pass when max_seconds is overridden to e.g. 30m (1800s)."""
        # 25 minutes = 1500s
        assert check_duration_limit(1500, max_seconds=1800) is False
        validate_video_duration(1500, max_seconds=1800)

    def test_duration_override_custom_limit_exceeded(self):
        """Videos exceeding the overridden limit still raise HTTP 400."""
        # 35 minutes = 2100s, limit = 30m (1800s)
        assert check_duration_limit(2100, max_seconds=1800) is True
        with pytest.raises(GuardrailViolationError) as exc_info:
            validate_video_duration(2100, max_seconds=1800)
        assert exc_info.value.status_code == 400
        assert "exceeds the strict operational limit of 1800 seconds (30 minutes)" in exc_info.value.detail

    def test_duration_override_unlimited(self):
        """Setting max_seconds=0 allows any video duration through (unlimited)."""
        # 2 hours = 7200s
        assert check_duration_limit(7200, max_seconds=0) is False
        validate_video_duration(7200, max_seconds=0)

    def test_duration_formatted(self):
        """Test formatting of durations."""
        assert format_duration(59) == "00:59"
        assert format_duration(600) == "10:00"
        assert format_duration(1200) == "20:00"
        assert format_duration(3665) == "01:01:05"
        assert format_duration(None) == "00:00"


class TestAudioFileSizeGuardrail:
    """Tests for the Audio File Size Guard (<= 25 MB)."""

    def test_audio_under_25mb_passes(self, tmp_path):
        """An audio file under 25 MB should pass validation."""
        audio_file = tmp_path / "valid_audio.mp3"
        # 5 MB file
        audio_file.write_bytes(b"\x00" * (5 * 1024 * 1024))

        size = validate_audio_file_size(audio_file, max_bytes=settings.MAX_AUDIO_SIZE_BYTES)
        assert size == 5 * 1024 * 1024
        assert audio_file.exists()

    def test_audio_exactly_25mb_passes(self, tmp_path):
        """An audio file of exactly 25 MB (26,214,400 bytes) should pass validation."""
        audio_file = tmp_path / "exact_audio.mp3"
        # 25 MB = 26,214,400 bytes
        max_bytes = 25 * 1024 * 1024
        audio_file.write_bytes(b"\x00" * max_bytes)

        size = validate_audio_file_size(audio_file, max_bytes=max_bytes)
        assert size == max_bytes
        assert audio_file.exists()

    def test_audio_exceeding_25mb_raises_http_400_and_purges_artifact(self, tmp_path):
        """An audio file exceeding 25 MB (26,214,400 bytes) must fail with HTTP 400 and be purged."""
        audio_file = tmp_path / "oversized_audio.mp3"
        # 25 MB + 1 byte
        max_bytes = 25 * 1024 * 1024
        audio_file.write_bytes(b"\x00" * (max_bytes + 1))

        with pytest.raises(GuardrailViolationError) as exc_info:
            validate_audio_file_size(audio_file, max_bytes=max_bytes)

        assert exc_info.value.status_code == 400
        assert "exceeds the strict operational limit of 25.00 MB" in exc_info.value.detail
        # Oversized file must be purged from disk
        assert not audio_file.exists()

    def test_missing_audio_raises_404(self, tmp_path):
        """Missing audio file should raise 404."""
        missing = tmp_path / "nonexistent.mp3"
        with pytest.raises(HTTPException) as exc_info:
            validate_audio_file_size(missing)
        assert exc_info.value.status_code == 404
