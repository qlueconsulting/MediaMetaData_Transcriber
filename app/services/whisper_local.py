"""Local GPU-accelerated Speech-to-Text service using faster-whisper.

Optimized for NVIDIA RTX 2060 Super 8GB VRAM with float16 compute type.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from app.config import settings
from app.utils.logger import log
from app.models.transcript import TranscriptSegment, TranscriptData, WordTimestamp


class LocalWhisperService:
    """Manages faster-whisper model loading and GPU inference across single or dual GPU setups."""

    def __init__(self):
        self._model = None
        self._batched_model = None
        self._loaded_model_name: Optional[str] = None
        self._device_used: Optional[str] = None
        self._loaded_device_index = None
        self._loaded_compute_type: Optional[str] = None

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

    def resolve_device_indices(self) -> Union[int, List[int]]:
        """Resolve GPU device indices from settings ('auto', 'all', '0,1', or int)."""
        if settings.WHISPER_DEVICE.lower() != "cuda":
            return 0

        cuda_count = 0
        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception:
            pass

        if cuda_count <= 0:
            return 0

        raw = settings.WHISPER_DEVICE_INDEX
        if isinstance(raw, list):
            valid = [i for i in raw if 0 <= i < cuda_count]
            return valid if len(valid) > 1 else (valid[0] if valid else 0)

        raw_str = str(raw).strip().lower()
        if raw_str in ("auto", "all"):
            if cuda_count > 1:
                return list(range(cuda_count))
            return 0

        if "," in raw_str:
            indices = [int(x.strip()) for x in raw_str.split(",") if x.strip().isdigit()]
            valid = [i for i in indices if 0 <= i < cuda_count]
            if len(valid) > 1:
                return valid
            if len(valid) == 1:
                return valid[0]
            return 0

        try:
            val = int(raw_str)
            return val if 0 <= val < cuda_count else 0
        except ValueError:
            return 0

    def get_gpu_info(self) -> Dict[str, Any]:
        """Query GPU memory and device details across all available server GPUs."""
        devices = []
        cuda_count = 0
        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception:
            pass

        # Real-time multi-GPU telemetry via nvidia-smi if available
        try:
            import subprocess
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        devices.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_total_mb": float(parts[2]),
                            "vram_free_mb": float(parts[3]),
                            "vram_used_mb": float(parts[4]),
                        })
        except Exception:
            pass

        active_indices = self.resolve_device_indices()
        cuda_available = len(devices) > 0 or cuda_count > 0

        primary_device = devices[0] if devices else {}
        total_vram = sum(d["vram_total_mb"] for d in devices) if devices else None
        free_vram = sum(d["vram_free_mb"] for d in devices) if devices else None

        return {
            "cuda_available": cuda_available,
            "device_count": len(devices) if devices else cuda_count,
            "device_name": primary_device.get("name") or (f"{cuda_count} CUDA Device(s)" if cuda_count else "None"),
            "vram_total_mb": primary_device.get("vram_total_mb"),
            "vram_free_mb": primary_device.get("vram_free_mb"),
            "total_cluster_vram_mb": total_vram,
            "total_cluster_free_vram_mb": free_vram,
            "devices": devices,
            "active_device_indices": active_indices,
            "details": f"Multi-GPU enabled: {len(devices) or cuda_count} device(s) detected; Whisper bound to index {active_indices}" if cuda_available else "No CUDA devices detected",
        }

    def load_model(
        self,
        force_reload: bool = False,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        """Lazy load or return the preloaded faster-whisper model spanning available GPUs."""
        from faster_whisper import WhisperModel

        target_model = model_name or settings.WHISPER_MODEL
        target_device = device or settings.WHISPER_DEVICE
        target_compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        device_indices = self.resolve_device_indices() if target_device == "cuda" else 0
        num_workers = len(device_indices) if isinstance(device_indices, list) else 1

        # Validate CUDA availability
        cuda_ok, cuda_msg = self.is_cuda_available()
        if target_device.lower() == "cuda" and not cuda_ok:
            log.warning(f"CUDA requested but unavailable ({cuda_msg}). Falling back to CPU for local whisper.")
            target_device = "cpu"
            target_compute_type = "int8" if target_compute_type == "float16" else target_compute_type
            device_indices = 0
            num_workers = 1

        if self._model is not None and not force_reload:
            if (
                self._loaded_model_name == target_model
                and self._device_used == target_device
                and self._loaded_device_index == device_indices
                and getattr(self, "_loaded_compute_type", None) == target_compute_type
            ):
                return self._model

        log.info(
            f"Loading faster-whisper model '{target_model}' "
            f"(device={target_device}, device_index={device_indices}, num_workers={num_workers}, "
            f"compute_type={target_compute_type}, download_root={settings.MODEL_DIR})"
        )

        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self._model = WhisperModel(
            model_size_or_path=target_model,
            device=target_device,
            device_index=device_indices,
            compute_type=target_compute_type,
            download_root=str(settings.MODEL_DIR),
            num_workers=num_workers,
        )
        self._loaded_model_name = target_model
        self._device_used = target_device
        self._loaded_device_index = device_indices
        self._loaded_compute_type = target_compute_type
        self._batched_model = None
        log.info(f"Loaded faster-whisper model '{target_model}' on {target_device} (indices={device_indices}, compute={target_compute_type}) successfully.")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        job_id: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = False,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        audio_duration: Optional[float] = None,
    ) -> TranscriptData:
        """Transcribe audio with safe batch sizing and automatic self-healing OOM recovery."""
        target_model = model_name or settings.WHISPER_MODEL
        model = self.load_model(model_name=target_model)
        log.info(f"Starting local transcription for job {job_id} using model {self._loaded_model_name} on {self._device_used}")

        # Safe batch sizing (cap at 8 on 8GB VRAM to avoid CUDA OOM)
        requested_batch_size = batch_size or settings.WHISPER_BATCH_SIZE
        effective_batch_size = min(requested_batch_size, 8) if requested_batch_size > 0 else 8

        def _drain_segments(generator) -> Tuple[List[TranscriptSegment], str]:
            """Safely iterate generator to trigger inference and catch runtime memory exceptions."""
            segs: List[TranscriptSegment] = []
            parts: List[str] = []
            for seg in generator:
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
                segs.append(segment_model)
                parts.append(seg.text)
            return segs, "".join(parts).strip()

        segments: Optional[List[TranscriptSegment]] = None
        full_text: Optional[str] = None
        info = None
        used_engine = f"faster-whisper-{self._device_used or 'cuda'}"

        # ----------------------------------------------------------------------
        # Strategy 1: BatchedInferencePipeline (Adaptive batch size with retry)
        # ----------------------------------------------------------------------
        if effective_batch_size > 1 and self._device_used == "cuda":
            for try_batch in [effective_batch_size, 4]:
                try:
                    from faster_whisper import BatchedInferencePipeline
                    if self._batched_model is None or getattr(self._batched_model, "model", None) != model:
                        self._batched_model = BatchedInferencePipeline(model=model)
                    log.info(
                        f"Executing BatchedInferencePipeline (model={target_model}, batch_size={try_batch}, "
                        f"beam_size={settings.WHISPER_BEAM_SIZE}, vad_filter={settings.WHISPER_VAD_FILTER})"
                    )
                    gen, current_info = self._batched_model.transcribe(
                        str(audio_path),
                        batch_size=try_batch,
                        beam_size=settings.WHISPER_BEAM_SIZE,
                        language=language,
                        initial_prompt=prompt,
                        word_timestamps=word_timestamps,
                        vad_filter=settings.WHISPER_VAD_FILTER,
                        chunk_length=30,
                    )
                    segments, full_text = _drain_segments(gen)
                    info = current_info
                    break  # Success
                except Exception as e:
                    err = str(e).lower()
                    if ("out of memory" in err or "cuda" in err) and try_batch > 4:
                        log.warning(f"Batched inference OOM at batch_size={try_batch}. Retrying with batch_size=4...")
                        continue
                    log.warning(f"Batched inference failed ({e}), falling back to sequential GPU transcribe")
                    break

        # ----------------------------------------------------------------------
        # Strategy 2: Standard sequential GPU transcribe (batch_size=1)
        # ----------------------------------------------------------------------
        if segments is None:
            try:
                log.info(f"Executing standard sequential transcribe for job {job_id}")
                gen, info = model.transcribe(
                    str(audio_path),
                    beam_size=settings.WHISPER_BEAM_SIZE,
                    language=language,
                    initial_prompt=prompt,
                    word_timestamps=word_timestamps,
                    vad_filter=settings.WHISPER_VAD_FILTER,
                )
                segments, full_text = _drain_segments(gen)
            except Exception as e:
                err = str(e).lower()
                log.warning(f"Standard sequential GPU transcribe failed: {e}")

                # --------------------------------------------------------------
                # Strategy 3: Reload with int8_float16 compute_type (cuts VRAM in half)
                # --------------------------------------------------------------
                if ("out of memory" in err or "cuda" in err) and getattr(self, "_loaded_compute_type", None) != "int8_float16":
                    try:
                        log.warning("CUDA OOM encountered: Auto-recovering with compute_type='int8_float16'...")
                        model = self.load_model(force_reload=True, model_name=target_model, compute_type="int8_float16")
                        gen, info = model.transcribe(
                            str(audio_path),
                            beam_size=settings.WHISPER_BEAM_SIZE,
                            language=language,
                            initial_prompt=prompt,
                            word_timestamps=word_timestamps,
                            vad_filter=settings.WHISPER_VAD_FILTER,
                        )
                        segments, full_text = _drain_segments(gen)
                    except Exception as e2:
                        log.warning(f"int8_float16 recovery failed: {e2}")

        # ----------------------------------------------------------------------
        # Strategy 4: Graceful CPU Fallback (Guarantees request never hard-fails)
        # ----------------------------------------------------------------------
        if segments is None:
            log.warning(f"All GPU execution strategies failed for job {job_id}. Falling back to CPU...")
            cpu_model = self.load_model(force_reload=True, model_name=target_model, device="cpu", compute_type="int8")
            gen, info = cpu_model.transcribe(
                str(audio_path),
                beam_size=settings.WHISPER_BEAM_SIZE,
                language=language,
                initial_prompt=prompt,
                word_timestamps=word_timestamps,
                vad_filter=settings.WHISPER_VAD_FILTER,
            )
            segments, full_text = _drain_segments(gen)
            used_engine = "faster-whisper-cpu"

        log.info(
            f"Local transcription finished for job {job_id}: "
            f"{len(segments)} segments, detected language '{getattr(info, 'language', 'unknown')}' "
            f"using {used_engine}"
        )

        return TranscriptData(
            job_id=job_id,
            text=full_text,
            language=getattr(info, "language", None),
            language_probability=round(info.language_probability, 4) if hasattr(info, "language_probability") and info.language_probability else None,
            duration_seconds=round(info.duration, 2) if hasattr(info, "duration") and info.duration else None,
            engine=used_engine,
            model=self._loaded_model_name or settings.WHISPER_MODEL,
            segments=segments,
        )


# Singleton instance
local_whisper_service = LocalWhisperService()
