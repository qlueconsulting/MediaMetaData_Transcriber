"""Unit tests for the transcription orchestrator and Groq Whisper fallback."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi import HTTPException

from app.config import settings
from app.models.transcript import TranscriptData, TranscriptSegment
from app.services.transcription import TranscriptionOrchestrator
from app.services.groq_client import GroqWhisperService


@pytest.fixture
def dummy_audio_file(tmp_path):
    """Create a dummy audio file."""
    audio = tmp_path / "test_audio.mp3"
    audio.write_bytes(b"\xFF\xFB\x90\x00" * 100)  # Dummy MP3 frame bytes
    return audio


def test_groq_service_not_configured_raises(dummy_audio_file, monkeypatch):
    """Groq service should raise 503 if GROQ_API_KEY is not set."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    service = GroqWhisperService()

    with pytest.raises(HTTPException) as exc:
        service.transcribe(dummy_audio_file, job_id="groq-job-1")
    assert exc.value.status_code == 503
    assert "GROQ_API_KEY is missing" in exc.value.detail


def test_orchestrator_cuda_to_groq_fallback(dummy_audio_file, monkeypatch):
    """When local faster-whisper fails (e.g. CUDA OOM or error), Groq fallback should execute."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_mock_key_123")
    monkeypatch.setattr(settings, "ENABLE_GROQ_FALLBACK", True)

    groq_transcript = TranscriptData(
        job_id="fallback-job-1",
        text="Fallback transcription from Groq.",
        language="en",
        duration_seconds=12.5,
        engine="groq-whisper-cloud",
        model="whisper-large-v3-turbo",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=5.0,
                text="Fallback transcription from Groq.",
            )
        ],
    )

    orchestrator = TranscriptionOrchestrator()

    # Local whisper throws RuntimeError (simulating CUDA OOM or missing GPU)
    with patch("app.services.whisper_local.local_whisper_service.transcribe", side_effect=RuntimeError("CUDA out of memory")), \
         patch("app.services.groq_client.groq_whisper_service.transcribe", return_value=groq_transcript) as mock_groq:

        result = orchestrator.transcribe(
            audio_path=dummy_audio_file,
            job_id="fallback-job-1",
        )

        assert result is not None
        assert result.engine == "groq-whisper-cloud"
        assert result.text == "Fallback transcription from Groq."
        mock_groq.assert_called_once()


def test_orchestrator_force_groq(dummy_audio_file, monkeypatch):
    """Explicitly requesting force_engine='groq' calls Groq directly."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_mock_key_123")

    groq_transcript = TranscriptData(
        job_id="force-groq-job",
        text="Forced Groq result.",
        language="en",
        duration_seconds=5.0,
        engine="groq-whisper-cloud",
        model="whisper-large-v3-turbo",
        segments=[],
    )

    orchestrator = TranscriptionOrchestrator()

    with patch("app.services.whisper_local.local_whisper_service.transcribe") as mock_local, \
         patch("app.services.groq_client.groq_whisper_service.transcribe", return_value=groq_transcript) as mock_groq:

        result = orchestrator.transcribe(
            audio_path=dummy_audio_file,
            job_id="force-groq-job",
            force_engine="groq",
        )

        assert result.text == "Forced Groq result."
        mock_groq.assert_called_once()
        mock_local.assert_not_called()


def test_orchestrator_adaptive_sla_routes_to_groq_for_very_long_audio(dummy_audio_file, monkeypatch):
    """When speed_profile='adaptive' and video > 30 mins (1800s), route to Groq for instant SLA if configured."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_mock_key_123")

    groq_transcript = TranscriptData(
        job_id="sla-groq-job",
        text="Fast cloud transcription for long video.",
        language="en",
        duration_seconds=2400.0,
        engine="groq-whisper-cloud",
        model="whisper-large-v3-turbo",
        segments=[],
    )

    orchestrator = TranscriptionOrchestrator()

    with patch("app.services.whisper_local.local_whisper_service.transcribe") as mock_local, \
         patch("app.services.groq_client.groq_whisper_service.transcribe", return_value=groq_transcript) as mock_groq:

        result = orchestrator.transcribe(
            audio_path=dummy_audio_file,
            job_id="sla-groq-job",
            speed_profile="adaptive",
            audio_duration=2400.0,  # 40-minute video
        )

        assert result.engine == "groq-whisper-cloud"
        mock_groq.assert_called_once()
        mock_local.assert_not_called()


def test_orchestrator_adaptive_sla_routes_to_local_turbo_for_standard_lengths(dummy_audio_file):
    """Adaptive SLA routes 20-min video to large-v3-turbo with dynamic batching on local GPU."""
    local_transcript = TranscriptData(
        job_id="local-turbo-job",
        text="Fast local turbo transcription.",
        language="en",
        duration_seconds=1200.0,
        engine="faster-whisper-cuda",
        model="large-v3-turbo",
        segments=[],
    )

    orchestrator = TranscriptionOrchestrator()

    with patch("app.services.whisper_local.local_whisper_service.transcribe", return_value=local_transcript) as mock_local:
        result = orchestrator.transcribe(
            audio_path=dummy_audio_file,
            job_id="local-turbo-job",
            speed_profile="adaptive",
            audio_duration=1200.0,
        )

        assert result.model == "large-v3-turbo"
        mock_local.assert_called_once()
        call_kwargs = mock_local.call_args.kwargs
        assert call_kwargs["model_name"] == "large-v3-turbo"
        assert call_kwargs["audio_duration"] == 1200.0
