"""Local GPU-accelerated Speech-to-Text service using faster-whisper.

Optimized for NVIDIA RTX 2060 Super 8GB VRAM with float16 compute type.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from app.config import settings
from app.utils.logger import log
from app.models.transcript import TranscriptSegment, TranscriptData, WordTimestamp


class LocalWhisperService:
    """Manages faster-whisper model loading and GPU inference."""

    def __init__(self):
        self._model = None
        self._loaded_model_name: Optional[str] = None
        self._device_used: Optional[str] = None

    def is_cuda_available(self) -> Tuple[bool, str]:
        """Check if CUDA acceleration is available on the host system."""
        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count()
            if cuda_count > 0:
                return True, f"Found {cuda_count} CUDA device(s)"
        except Exception as e:
            log.debug(f"ctranslate2 CUDA check error: {e}")

        # Secondary check with PyTorch if available
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                return True, f"CUDA available: {device_name}"
        except Exception as e:
            log.debug(f"torch CUDA check error: {e}")

        return False, "No CUDA devices detected"

    def get_gpu_info(self) -> Dict[str, Any]:
        """Query GPU memory and device details."""
        info = {
            "cuda_available": False,
            "device_name": None,
            "vram_total_mb": None,
            "vram_allocated_mb": None,
            "vram_free_mb": None,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                info["vram_total_mb"] = round(total, 2)
                info["vram_allocated_mb"] = round(allocated, 2)
                info["vram_free_mb"] = round(total - allocated, 2)
        except Exception:
            pass

        if not info["cuda_available"]:
            cuda_ok, msg = self.is_cuda_available()
            info["cuda_available"] = cuda_ok
            info["details"] = msg

        return info

    def load_model(self, force_reload: bool = False):
        """Lazy load or return the preloaded faster-whisper model."""
        from faster_whisper import WhisperModel

        target_model = settings.WHISPER_MODEL
        target_device = settings.WHISPER_DEVICE
        compute_type = settings.WHISPER_COMPUTE_TYPE

        # Validate CUDA availability
        cuda_ok, cuda_msg = self.is_cuda_available()
        if target_device.lower() == "cuda" and not cuda_ok:
            log.warning(f"CUDA requested but unavailable ({cuda_msg}). Falling back to CPU for local whisper.")
            target_device = "cpu"
            compute_type = "int8" if compute_type == "float16" else compute_type

        if self._model is not None and not force_reload:
            if self._loaded_model_name == target_model and self._device_used == target_device:
                return self._model

        log.info(
            f"Loading faster-whisper model '{target_model}' "
            f"(device={target_device}, compute_type={compute_type}, download_root={settings.MODEL_DIR})"
        )

        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self._model = WhisperModel(
            model_size_or_path=target_model,
            device=target_device,
            device_index=settings.WHISPER_DEVICE_INDEX,
            compute_type=compute_type,
            download_root=str(settings.MODEL_DIR),
        )
        self._loaded_model_name = target_model
        self._device_used = target_device
        log.info(f"Loaded faster-whisper model '{target_model}' on {target_device} successfully.")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        job_id: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptData:
        """Transcribe audio file using local GPU faster-whisper."""
        model = self.load_model()
        log.info(f"Starting local GPU transcription for job {job_id} using model {self._loaded_model_name}")

        segments_gen, info = model.transcribe(
            str(audio_path),
            beam_size=settings.WHISPER_BEAM_SIZE,
            language=language,
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
        )

        segments: List[TranscriptSegment] = []
        full_text_parts: List[str] = []

        for seg in segments_gen:
            words = None
            if word_timestamps and hasattr(seg, "words") and seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=getattr(w, "probability", None),
                    )
                    for w in seg.words
                ]

            segment_model = TranscriptSegment(
                id=seg.id,
                seek=getattr(seg, "seek", None),
                start=seg.start,
                end=seg.end,
                text=seg.text,
                avg_logprob=getattr(seg, "avg_logprob", None),
                no_speech_prob=getattr(seg, "no_speech_prob", None),
                words=words,
            )
            segments.append(segment_model)
            full_text_parts.append(seg.text)

        full_text = "".join(full_text_parts).strip()
        engine_name = f"faster-whisper-{self._device_used or 'cuda'}"

        log.info(
            f"Local transcription finished for job {job_id}: "
            f"{len(segments)} segments, detected language '{info.language}' (prob={info.language_probability:.2f})"
        )

        return TranscriptData(
            job_id=job_id,
            text=full_text,
            language=info.language,
            language_probability=round(info.language_probability, 4) if info.language_probability else None,
            duration_seconds=round(info.duration, 2) if hasattr(info, "duration") and info.duration else None,
            engine=engine_name,
            model=self._loaded_model_name or settings.WHISPER_MODEL,
            segments=segments,
        )


# Singleton instance
local_whisper_service = LocalWhisperService()
