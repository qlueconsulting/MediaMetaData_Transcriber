"""Integration and API route tests for MediaMetaData_Transcriber."""

from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.models.metadata import MediaMetadataResponse
from app.models.transcript import TranscriptData, TranscriptSegment


class TestHealthEndpoints:
    """Test health check and root endpoints."""

    def test_root_endpoint_json(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "MediaMetaData_Transcriber"
        assert data["status"] == "online"
        assert data["guardrails"]["max_video_duration_seconds"] == 1200

    def test_root_endpoint_html(self, client: TestClient):
        response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MediaMetaData_Transcriber | Testing Apparatus" in response.text

    def test_testing_ui_endpoint(self, client: TestClient):
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MediaMetaData_Transcriber | Testing Apparatus" in response.text

    def test_test_endpoint(self, client: TestClient):
        response = client.get("/test")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Target Media Input" in response.text

    def test_health_endpoint(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "hardware" in data
        assert "whisper" in data
        assert "guardrails" in data
        assert data["guardrails"]["max_duration_seconds"] == 1200


class TestMetadataEndpoint:
    """Test metadata extraction endpoint with the early 20-minute decision flag."""

    def test_metadata_under_20_minutes(self, client: TestClient, sample_meta_dict):
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_inst = MagicMock()
            mock_inst.extract_info.return_value = sample_meta_dict
            mock_ydl.return_value.__enter__.return_value = mock_inst

            response = client.post(
                "/api/v1/metadata",
                json={"url": "https://www.youtube.com/watch?v=valid123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Introduction to AI Audio Processing"
            assert data["duration_seconds"] == 600
            assert data["exceeds_duration_limit"] is False
            assert data["allowed_for_transcription"] is True
            assert data["warning"] is None

    def test_metadata_over_20_minutes_is_flagged(self, client: TestClient, sample_oversized_meta_dict):
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_inst = MagicMock()
            mock_inst.extract_info.return_value = sample_oversized_meta_dict
            mock_ydl.return_value.__enter__.return_value = mock_inst

            response = client.post(
                "/api/v1/metadata",
                json={"url": "https://www.youtube.com/watch?v=longpodcast"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Long 25-Minute Podcast Episode"
            assert data["duration_seconds"] == 1500
            assert data["exceeds_duration_limit"] is True
            assert data["allowed_for_transcription"] is False
            assert "exceeds the strict limit of 1200 seconds" in data["warning"]


class TestTranscriptionAndAudioExtractionGuardrails:
    """Test transcribe & extract endpoints enforcing the 20-min early rejection and 25MB limits."""

    def test_transcribe_over_20_minutes_rejected_with_http_400(self, client: TestClient, sample_oversized_meta_dict):
        """Any attempt to trigger transcription on video > 20 minutes MUST be rejected with HTTP 400."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_inst = MagicMock()
            mock_inst.extract_info.return_value = sample_oversized_meta_dict
            mock_ydl.return_value.__enter__.return_value = mock_inst

            response = client.post(
                "/api/v1/transcribe",
                json={"url": "https://www.youtube.com/watch?v=longpodcast"},
            )

            assert response.status_code == 400
            assert "exceeds the operational limit of 1200 seconds (20 minutes)" in response.json()["detail"]

    def test_extract_audio_over_20_minutes_rejected_with_http_400(self, client: TestClient, sample_oversized_meta_dict):
        """Any attempt to trigger audio extraction on video > 20 minutes MUST be rejected with HTTP 400."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_inst = MagicMock()
            mock_inst.extract_info.return_value = sample_oversized_meta_dict
            mock_ydl.return_value.__enter__.return_value = mock_inst

            response = client.post(
                "/api/v1/extract-audio",
                json={"url": "https://www.youtube.com/watch?v=longpodcast"},
            )

            assert response.status_code == 400
            assert "exceeds the operational limit of 1200 seconds (20 minutes)" in response.json()["detail"]

    def test_transcribe_pipeline_success(self, client: TestClient, sample_meta_dict, tmp_path):
        """Test full transcription pipeline on valid video."""
        sample_meta_resp = MediaMetadataResponse(
            url="https://www.youtube.com/watch?v=valid123",
            title="Sample Title",
            creator="Author",
            duration_seconds=300,
            duration_formatted="05:00",
            platform="youtube",
            exceeds_duration_limit=False,
            allowed_for_transcription=True,
        )

        mock_transcript = TranscriptData(
            job_id="dummy-id",
            text="Hello world, this is a test audio transcription.",
            language="en",
            duration_seconds=300.0,
            engine="faster-whisper-cuda",
            model="large-v3",
            segments=[
                TranscriptSegment(
                    id=0,
                    start=0.0,
                    end=4.0,
                    text="Hello world,",
                ),
                TranscriptSegment(
                    id=1,
                    start=4.0,
                    end=8.0,
                    text=" this is a test audio transcription.",
                ),
            ],
        )

        with patch("app.services.media.media_service.extract_metadata", return_value=sample_meta_resp), \
             patch("app.services.media.media_service.download_and_extract_audio") as mock_dl, \
             patch("app.services.transcription.transcription_orchestrator.transcribe", return_value=mock_transcript):

            def fake_dl(url, job_id, pre_extracted_meta=None):
                from app.services.storage import storage_service
                audio_file = storage_service.get_audio_path(job_id)
                audio_file.parent.mkdir(parents=True, exist_ok=True)
                audio_file.write_bytes(b"\x00" * 5000)
                storage_service.save_meta(job_id, sample_meta_resp.model_dump())
                return audio_file, sample_meta_resp

            mock_dl.side_effect = fake_dl

            response = client.post(
                "/api/v1/transcribe",
                json={"url": "https://www.youtube.com/watch?v=valid123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["audio_file_available"] is True
            assert data["transcript"]["text"] == "Hello world, this is a test audio transcription."
            job_id = data["job_id"]

            # Test artifact retrieval endpoints
            meta_res = client.get(f"/api/v1/jobs/{job_id}/meta")
            assert meta_res.status_code == 200

            audio_res = client.get(f"/api/v1/jobs/{job_id}/audio")
            assert audio_res.status_code == 200
            assert audio_res.headers["content-type"] == "audio/mpeg"

            # Test transcript formats
            # 1. JSON
            tr_json = client.get(f"/api/v1/jobs/{job_id}/transcript?format=json")
            assert tr_json.status_code == 200
            assert tr_json.json()["language"] == "en"

            # 2. Plain Text
            tr_txt = client.get(f"/api/v1/jobs/{job_id}/transcript?format=text")
            assert tr_txt.status_code == 200
            assert "Hello world, this is a test audio transcription." in tr_txt.text

            # 3. SubRip (SRT)
            tr_srt = client.get(f"/api/v1/jobs/{job_id}/transcript?format=srt")
            assert tr_srt.status_code == 200
            assert "00:00:00,000 --> 00:00:04,000" in tr_srt.text

            # 4. WebVTT (VTT)
            tr_vtt = client.get(f"/api/v1/jobs/{job_id}/transcript?format=vtt")
            assert tr_vtt.status_code == 200
            assert "WEBVTT" in tr_vtt.text
            assert "00:00:00.000 --> 00:00:04.000" in tr_vtt.text
