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

    def test_metadata_over_20_minutes_allowed_with_override(self, client: TestClient, sample_oversized_meta_dict):
        """Videos > 20m pass metadata guardrail when max_duration_minutes override is provided."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_inst = MagicMock()
            mock_inst.extract_info.return_value = sample_oversized_meta_dict
            mock_ydl.return_value.__enter__.return_value = mock_inst

            response = client.post(
                "/api/v1/metadata",
                json={
                    "url": "https://www.youtube.com/watch?v=longpodcast",
                    "max_duration_minutes": 30,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["duration_seconds"] == 1500
            assert data["max_duration_seconds"] == 1800
            assert data["exceeds_duration_limit"] is False
            assert data["allowed_for_transcription"] is True
            assert data["warning"] is None


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

    def test_transcribe_over_20_minutes_allowed_with_override(self, client: TestClient, sample_oversized_meta_dict):
        """When max_duration_minutes override is supplied, video > 20m proceeds without guardrail rejection."""
        oversized_meta_resp = MediaMetadataResponse(
            url="https://www.youtube.com/watch?v=longpodcast",
            title="Long 25-Minute Podcast Episode",
            creator="PodcastStudio",
            duration_seconds=1500,
            duration_formatted="25:00",
            platform="youtube",
            exceeds_duration_limit=False,
            max_duration_seconds=1800,
            allowed_for_transcription=True,
        )

        mock_transcript = TranscriptData(
            job_id="dummy-overridden-id",
            text="This is a 25 minute podcast transcribed.",
            language="en",
            duration_seconds=1500.0,
            engine="faster-whisper-cuda",
            model="large-v3",
            segments=[],
        )

        with patch("app.services.media.media_service.extract_metadata", return_value=oversized_meta_resp), \
             patch("app.services.media.media_service.download_and_extract_audio") as mock_dl, \
             patch("app.services.transcription.transcription_orchestrator.transcribe", return_value=mock_transcript):

            def fake_dl(url, job_id, pre_extracted_meta=None, **kwargs):
                from app.services.storage import storage_service
                audio_file = storage_service.get_audio_path(job_id)
                audio_file.parent.mkdir(parents=True, exist_ok=True)
                audio_file.write_bytes(b"\x00" * 5000)
                storage_service.save_meta(job_id, oversized_meta_resp.model_dump())
                return audio_file, oversized_meta_resp

            mock_dl.side_effect = fake_dl

            response = client.post(
                "/api/v1/transcribe",
                json={
                    "url": "https://www.youtube.com/watch?v=longpodcast",
                    "max_duration_minutes": 30,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["meta"]["duration_seconds"] == 1500
            assert data["meta"]["max_duration_seconds"] == 1800

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

    def test_metadata_caching_and_bypass(self, client: TestClient):
        """Test metadata caching and bypass_cache flag behavior."""
        from app.services.storage import storage_service

        job_id = "cached-meta-job"
        test_url = "https://www.youtube.com/watch?v=cached123"
        storage_service.save_meta(job_id, {
            "url": test_url,
            "title": "Cached Title",
            "creator": "Cached Creator",
            "duration_seconds": 120,
            "duration_formatted": "02:00",
            "platform": "youtube",
            "exceeds_duration_limit": False,
            "allowed_for_transcription": True,
        })

        # By default bypass_cache is False -> returns cached metadata
        res = client.post("/api/v1/metadata", json={"url": test_url})
        assert res.status_code == 200
        data = res.json()
        assert data["title"] == "Cached Title"
        assert data["cached"] is True
        assert data["cached_job_id"] == job_id

        # With bypass_cache=True, mock yt-dlp to prove it bypassed cache
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_inst = MagicMock()
            mock_inst.extract_info.return_value = {
                "id": "cached123",
                "title": "Fresh Title from YTDLP",
                "duration": 120,
                "extractor": "youtube",
            }
            mock_ydl.return_value.__enter__.return_value = mock_inst

            bypass_res = client.post("/api/v1/metadata", json={"url": test_url, "bypass_cache": True})
            assert bypass_res.status_code == 200
            bypass_data = bypass_res.json()
            assert bypass_data["title"] == "Fresh Title from YTDLP"
            assert bypass_data["cached"] is False

    def test_job_transcribe_endpoint_and_cache(self, client: TestClient):
        """Test POST /api/v1/jobs/{job_id}/transcribe for pre-extracted audio."""
        from app.services.storage import storage_service
        from app.models.transcript import TranscriptData, TranscriptSegment

        job_id = "modular-job-001"
        audio_file = storage_service.get_audio_path(job_id)
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"\x00" * 2048)

        storage_service.save_meta(job_id, {
            "url": "https://www.youtube.com/watch?v=modular123",
            "title": "Modular Audio Job",
            "duration_seconds": 60,
            "duration_formatted": "01:00",
            "exceeds_duration_limit": False,
            "allowed_for_transcription": True,
        })

        mock_transcript = TranscriptData(
            job_id=job_id,
            text="Modular transcription test passed.",
            language="en",
            duration_seconds=60.0,
            engine="faster-whisper-cuda",
            model="large-v3",
            segments=[
                TranscriptSegment(id=0, start=0.0, end=2.0, text="Modular transcription test passed.")
            ],
        )

        with patch("app.services.transcription.transcription_orchestrator.transcribe", return_value=mock_transcript):
            # 1. Transcribe the existing job
            res = client.post(f"/api/v1/jobs/{job_id}/transcribe", json={"bypass_cache": False})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert data["transcript"]["text"] == "Modular transcription test passed."
            assert data["cached"] is False

            # 2. Transcribe again with bypass_cache=False -> returns cached: True instantly
            res_cached = client.post(f"/api/v1/jobs/{job_id}/transcribe", json={"bypass_cache": False})
            assert res_cached.status_code == 200
            assert res_cached.json()["cached"] is True

            # 3. Requesting a nonexistent job returns 400
            res_missing = client.post("/api/v1/jobs/nonexistent-id/transcribe", json={})
            assert res_missing.status_code == 400

    def test_transcribe_cache_hit_and_bypass(self, client: TestClient):
        """Test POST /api/v1/transcribe returns cached result when available and bypasses when requested."""
        from app.services.storage import storage_service
        from app.models.transcript import TranscriptData, TranscriptSegment

        job_id = "cached-full-job"
        test_url = "https://www.youtube.com/watch?v=fullcache123"

        storage_service.save_meta(job_id, {
            "url": test_url,
            "title": "Full Cache Title",
            "duration_seconds": 90,
            "duration_formatted": "01:30",
            "exceeds_duration_limit": False,
            "allowed_for_transcription": True,
        })
        storage_service.save_transcript(job_id, {
            "job_id": job_id,
            "text": "Cached full transcription text.",
            "language": "en",
            "duration_seconds": 90.0,
            "engine": "faster-whisper-cuda",
            "model": "large-v3",
            "segments": [],
        })
        storage_service.get_audio_path(job_id).write_bytes(b"\x00" * 500)

        # 1. Without bypass_cache, returns cached immediately
        res = client.post("/api/v1/transcribe", json={"url": test_url, "bypass_cache": False})
        assert res.status_code == 200
        data = res.json()
        assert data["cached"] is True
        assert data["job_id"] == job_id
        assert data["transcript"]["text"] == "Cached full transcription text."

        # 2. With bypass_cache=True, executes freshly
        fresh_meta = MediaMetadataResponse(
            url=test_url,
            title="Fresh Live Title",
            duration_seconds=90,
            duration_formatted="01:30",
            exceeds_duration_limit=False,
            allowed_for_transcription=True,
        )
        fresh_transcript = TranscriptData(
            job_id="new-fresh-id",
            text="Freshly transcribed live text.",
            language="en",
            duration_seconds=90.0,
            engine="faster-whisper-cuda",
            model="large-v3",
            segments=[],
        )

        with patch("app.services.media.media_service.extract_metadata", return_value=fresh_meta), \
             patch("app.services.media.media_service.download_and_extract_audio") as mock_dl, \
             patch("app.services.transcription.transcription_orchestrator.transcribe", return_value=fresh_transcript):

            def fake_fresh_dl(url, job_id, pre_extracted_meta=None, **kwargs):
                audio_file = storage_service.get_audio_path(job_id)
                audio_file.parent.mkdir(parents=True, exist_ok=True)
                audio_file.write_bytes(b"\x00" * 600)
                storage_service.save_meta(job_id, fresh_meta.model_dump())
                return audio_file, fresh_meta

            mock_dl.side_effect = fake_fresh_dl

            bypass_res = client.post("/api/v1/transcribe", json={"url": test_url, "bypass_cache": True})
            assert bypass_res.status_code == 200
            bypass_data = bypass_res.json()
            assert bypass_data["cached"] is False
            assert bypass_data["transcript"]["text"] == "Freshly transcribed live text."


class TestApiKeyAndRouteCompatibility:
    """Test optional API_KEY enforcement and /api/* route aliasing."""

    def test_api_route_without_v1_prefix(self, client: TestClient, sample_meta_dict):
        """Routes mounted at /api/metadata should work identically to /api/v1/metadata."""
        with patch("app.services.media.media_service._get_base_ytdlp_opts", return_value={}), \
             patch("yt_dlp.YoutubeDL.extract_info", return_value=sample_meta_dict):
            response = client.post("/api/metadata", json={"url": "https://example.com/test"})
            assert response.status_code == 200
            assert response.json()["title"] == "Introduction to AI Audio Processing"

    def test_api_key_enforcement_when_configured(self, client: TestClient, monkeypatch, sample_meta_dict):
        """When settings.API_KEY is configured, unauthenticated requests must be rejected with 401."""
        from app.config import settings
        test_key = "test-secret-key-12345"
        monkeypatch.setattr(settings, "API_KEY", test_key)

        with patch("app.services.media.media_service._get_base_ytdlp_opts", return_value={}), \
             patch("yt_dlp.YoutubeDL.extract_info", return_value=sample_meta_dict):

            # 1. Request without key -> 401
            res_no_key = client.post("/api/v1/metadata", json={"url": "https://example.com/test"})
            assert res_no_key.status_code == 401
            assert "Unauthorized" in res_no_key.json()["detail"]

            # 2. Request with wrong key -> 401
            res_wrong_key = client.post(
                "/api/v1/metadata",
                json={"url": "https://example.com/test"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert res_wrong_key.status_code == 401

            # 3. Request with valid X-API-Key header -> 200
            res_valid_key = client.post(
                "/api/v1/metadata",
                json={"url": "https://example.com/test"},
                headers={"X-API-Key": test_key},
            )
            assert res_valid_key.status_code == 200

            # 4. Request with valid Authorization: Bearer token -> 200
            res_bearer = client.post(
                "/api/v1/metadata",
                json={"url": "https://example.com/test"},
                headers={"Authorization": f"Bearer {test_key}"},
            )
            assert res_bearer.status_code == 200

            # 5. Public health check remains accessible without key
            health_res = client.get("/api/v1/health")
            assert health_res.status_code == 200



