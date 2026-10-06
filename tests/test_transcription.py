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


def test_local_whisper_load_model_accepts_model_name(monkeypatch):
    """LocalWhisperService.load_model must accept model_name keyword argument."""
    from app.services.whisper_local import LocalWhisperService

    service = LocalWhisperService()
    mock_model_instance = MagicMock()

    with patch("faster_whisper.WhisperModel", return_value=mock_model_instance) as mock_cls:
        # Test call with model_name keyword argument
        loaded = service.load_model(model_name="large-v3-turbo")
        assert loaded == mock_model_instance
        assert service._loaded_model_name == "large-v3-turbo"
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model_size_or_path"] == "large-v3-turbo"


def test_local_whisper_transcribe_with_model_name(dummy_audio_file):
    """LocalWhisperService.transcribe must pass model_name to load_model without error."""
    from app.services.whisper_local import LocalWhisperService

    service = LocalWhisperService()
    mock_model_instance = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0
    mock_model_instance.transcribe.return_value = ([], mock_info)

    with patch.object(service, "load_model", return_value=mock_model_instance) as mock_load:
        # Ensure CUDA is mocked as false to test standard execution path safely
        with patch.object(service, "is_cuda_available", return_value=(False, "CPU mode")):
            result = service.transcribe(
                audio_path=dummy_audio_file,
                job_id="test-load-model-job",
                model_name="large-v3-turbo",
                audio_duration=15.0,
            )
            mock_load.assert_called_once_with(model_name="large-v3-turbo")
            assert result.job_id == "test-load-model-job"


def test_multi_gpu_device_resolution(monkeypatch):
    """Test resolution of GPU device indices for single, dual, auto, and manual configurations."""
    from app.services.whisper_local import LocalWhisperService

    service = LocalWhisperService()

    # Simulate 2 CUDA devices detected
    with patch("ctranslate2.get_cuda_device_count", return_value=2):
        monkeypatch.setattr(settings, "WHISPER_DEVICE", "cuda")

        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "auto")
        assert service.resolve_device_indices() == [0, 1]

        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "all")
        assert service.resolve_device_indices() == [0, 1]

        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "0,1")
        assert service.resolve_device_indices() == [0, 1]

        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "0")
        assert service.resolve_device_indices() == 0

        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "1")
        assert service.resolve_device_indices() == 1

    # Simulate 1 CUDA device
    with patch("ctranslate2.get_cuda_device_count", return_value=1):
        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "auto")
        assert service.resolve_device_indices() == 0


def test_multi_gpu_model_loading_spans_workers(monkeypatch):
    """When multiple GPUs are available, WhisperModel should be instantiated with device_index list and num_workers."""
    from app.services.whisper_local import LocalWhisperService

    service = LocalWhisperService()
    mock_model_instance = MagicMock()

    with patch("ctranslate2.get_cuda_device_count", return_value=2), \
         patch.object(service, "is_cuda_available", return_value=(True, "2 CUDA devices")), \
         patch("faster_whisper.WhisperModel", return_value=mock_model_instance) as mock_cls:
        monkeypatch.setattr(settings, "WHISPER_DEVICE", "cuda")
        monkeypatch.setattr(settings, "WHISPER_DEVICE_INDEX", "auto")

        loaded = service.load_model(force_reload=True)
        assert loaded == mock_model_instance
        assert mock_cls.call_args.kwargs["device_index"] == [0, 1]
        assert mock_cls.call_args.kwargs["num_workers"] == 2


def test_local_whisper_batched_oom_auto_recovery(dummy_audio_file):
    """When batched inference throws CUDA out of memory, it should fall back to sequential transcribe."""
    from app.services.whisper_local import LocalWhisperService

    service = LocalWhisperService()
    mock_model_instance = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    # Sequential transcribe returns valid segments
    mock_seg = MagicMock()
    mock_seg.id = 1
    mock_seg.seek = 0
    mock_seg.start = 0.0
    mock_seg.end = 2.0
    mock_seg.text = "Recovered after OOM."
    mock_seg.avg_logprob = -0.2
    mock_seg.no_speech_prob = 0.01
    mock_seg.words = None

    mock_model_instance.transcribe.return_value = ([mock_seg], mock_info)

    mock_batched_instance = MagicMock()
    # Batched transcribe generator throws CUDA out of memory on iteration
    def _oom_generator(*args, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")
        yield  # Make it a generator

    mock_batched_instance.transcribe.side_effect = _oom_generator

    with patch.object(service, "load_model", return_value=mock_model_instance), \
         patch.object(service, "is_cuda_available", return_value=(True, "CUDA available")), \
         patch("faster_whisper.BatchedInferencePipeline", return_value=mock_batched_instance):
        service._device_used = "cuda"

        result = service.transcribe(
            audio_path=dummy_audio_file,
            job_id="oom-recovery-job",
            batch_size=8,
        )

        assert result is not None
        assert result.text == "Recovered after OOM."
        assert len(result.segments) == 1
        mock_model_instance.transcribe.assert_called_once()


def test_local_whisper_complete_oom_falls_back_to_cpu(dummy_audio_file):
    """When all GPU transcribe strategies OOM, service should gracefully fall back to CPU."""
    from app.services.whisper_local import LocalWhisperService

    service = LocalWhisperService()

    gpu_model_instance = MagicMock()
    def _gpu_oom(*args, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")
    gpu_model_instance.transcribe.side_effect = _gpu_oom

    cpu_model_instance = MagicMock()
    mock_seg = MagicMock()
    mock_seg.id = 1
    mock_seg.seek = 0
    mock_seg.start = 0.0
    mock_seg.end = 2.0
    mock_seg.text = "CPU fallback success."
    mock_seg.avg_logprob = -0.2
    mock_seg.no_speech_prob = 0.01
    mock_seg.words = None
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0
    cpu_model_instance.transcribe.return_value = ([mock_seg], mock_info)

    def mock_load_model(force_reload=False, model_name=None, device=None, compute_type=None):
        if device == "cpu":
            return cpu_model_instance
        return gpu_model_instance

    with patch.object(service, "load_model", side_effect=mock_load_model), \
         patch.object(service, "is_cuda_available", return_value=(True, "CUDA available")):
        service._device_used = "cuda"

        result = service.transcribe(
            audio_path=dummy_audio_file,
            job_id="cpu-fallback-job",
            batch_size=1,  # Skip batched
        )

        assert result is not None
        assert result.text == "CPU fallback success."
        assert result.engine == "faster-whisper-cpu"



