"""Application configuration settings."""

from pathlib import Path
from typing import Optional, Union, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for MediaMetaData_Transcriber."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_KEY: Optional[str] = Field(
        default=None,
        description="Optional API key for securing endpoints. If configured, requests must provide X-API-Key header.",
    )

    # Persistent Storage Paths
    # Common persistent storage structure: /srv/storage/jobs/{job_id}/
    STORAGE_DIR: Path = Field(
        default=Path("/srv/storage/jobs"),
        description="Path to jobs root directory where /srv/storage/jobs/{job_id}/ is stored",
    )
    MODEL_DIR: Path = Field(
        default=Path("/srv/storage/models"),
        description="Path to cache directory for faster-whisper models",
    )

    # Strict Operational Guardrails
    # 1. 20-Minute (1200 seconds) ceiling
    MAX_DURATION_SECONDS: int = Field(
        default=1200,
        description="Strict maximum allowed video duration in seconds (20 minutes)",
    )
    # 2. 25 MB Audio File Size Guard (25 * 1024 * 1024 = 26,214,400 bytes)
    MAX_AUDIO_SIZE_BYTES: int = Field(
        default=26_214_400,
        description="Strict maximum allowed audio file size in bytes (25 MB)",
    )

    # Audio Encoding Specs (16kHz mono MP3)
    AUDIO_SAMPLE_RATE: int = Field(
        default=16000,
        description="Audio sampling rate in Hz (16kHz for optimal Whisper performance)",
    )
    AUDIO_CHANNELS: int = Field(
        default=1,
        description="Number of audio channels (1 for Mono)",
    )
    AUDIO_BITRATE: str = Field(
        default="64k",
        description="Audio bitrate (64k to minimize size while preserving speech clarity)",
    )

    # Local GPU Speech-to-Text (faster-whisper)
    WHISPER_MODEL: str = Field(
        default="large-v3-turbo",
        description="Whisper model name (default large-v3-turbo for 4x-6x speedup, large-v3, medium, etc.)",
    )
    WHISPER_DEVICE: str = Field(
        default="cuda",
        description="Computation device ('cuda' or 'cpu')",
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="float16",
        description="Computation type ('float16', 'int8_float16', 'int8')",
    )
    WHISPER_DEVICE_INDEX: Union[int, List[int], str] = Field(
        default="auto",
        description="GPU device index or indices ('auto' to span all available GPUs, 0, '0,1', or [0, 1])",
    )
    WHISPER_BEAM_SIZE: int = Field(
        default=1,
        description="Beam search size for decoding (1 = fast greedy search, 5 = exhaustive)",
    )
    WHISPER_BATCH_SIZE: int = Field(
        default=8,
        description="Batch size for BatchedInferencePipeline (8 optimal for 8GB VRAM saturation without OOM)",
    )
    WHISPER_VAD_FILTER: bool = Field(
        default=True,
        description="Enable Silero VAD to skip non-speech/silence sections, dramatically speeding up inference",
    )
    WHISPER_SLA_TARGET_SECONDS: int = Field(
        default=30,
        description="Target maximum pipeline execution time SLA in seconds (default: 30s)",
    )
    WHISPER_ADAPTIVE_SLA: bool = Field(
        default=True,
        description="Dynamically scale batch size and engine/model tier based on video duration to enforce < 30s SLA",
    )

    # Groq Cloud Whisper API Fallback
    ENABLE_GROQ_FALLBACK: bool = Field(
        default=True,
        description="Enable fallback to Groq Whisper API if GPU is unavailable or OOM",
    )
    FORCE_GROQ_FALLBACK: bool = Field(
        default=False,
        description="Force using Groq Whisper API instead of local GPU",
    )
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Groq Cloud",
    )
    GROQ_MODEL: str = Field(
        default="whisper-large-v3-turbo",
        description="Groq Whisper model identifier",
    )

    # yt-dlp Network / Residential Settings
    YTDLP_PLAYER_CLIENTS: str = Field(
        default="web,android",
        description="Comma-separated player clients to bypass YouTube 403/429 rate limits",
    )
    YTDLP_SOCKET_TIMEOUT: int = Field(
        default=30,
        description="Socket timeout in seconds for yt-dlp",
    )
    YTDLP_RETRIES: int = Field(
        default=5,
        description="Number of download retries for yt-dlp",
    )
    YTDLP_PROXY: Optional[str] = Field(
        default=None,
        description="Optional proxy URL for yt-dlp",
    )

    def ensure_directories(self) -> None:
        """Ensure persistent storage and model directories exist."""
        try:
            self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            # When running in test environments or non-root without /srv access,
            # fallback relative paths can be used if configured.
            pass

    @property
    def max_audio_size_mb(self) -> float:
        """Return max audio size in megabytes."""
        return self.MAX_AUDIO_SIZE_BYTES / (1024 * 1024)

    @property
    def max_duration_minutes(self) -> float:
        """Return max duration in minutes."""
        return self.MAX_DURATION_SECONDS / 60.0


settings = Settings()
