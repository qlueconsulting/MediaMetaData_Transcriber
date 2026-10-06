#!/usr/bin/env python3
"""Diagnostic script to verify CUDA, CTranslate2, and faster-whisper GPU capabilities."""

import sys


def check_cuda_environment():
    print("=" * 60)
    print("MediaMetaData_Transcriber: CUDA & Hardware Verification")
    print("=" * 60)

    # 1. Check CTranslate2
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        print(f"[ctranslate2] Version: {ctranslate2.__version__}")
        print(f"[ctranslate2] CUDA Devices Detected: {cuda_count}")
        if cuda_count > 0:
            print("  -> CTranslate2 GPU execution is ENABLED")
        else:
            print("  -> [WARNING] CTranslate2 detects 0 CUDA devices")
    except ImportError:
        print("[ctranslate2] NOT INSTALLED")

    # 2. Check PyTorch CUDA (if installed)
    try:
        import torch
        print(f"\n[torch] Version: {torch.__version__}")
        print(f"[torch] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[torch] Device Count: {torch.cuda.device_count()}")
            print(f"[torch] Device Name: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024 ** 3)
            print(f"[torch] Total VRAM: {total_vram_gb:.2f} GB")
    except ImportError:
        print("\n[torch] (Not installed in this environment)")

    # 3. Check faster-whisper
    try:
        import faster_whisper
        print(f"\n[faster-whisper] Version: {faster_whisper.__version__}")
    except ImportError:
        print("\n[faster-whisper] NOT INSTALLED")

    print("=" * 60)


if __name__ == "__main__":
    check_cuda_environment()
