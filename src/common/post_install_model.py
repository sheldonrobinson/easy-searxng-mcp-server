#!/usr/bin/env python3
"""Post-install script to download the faster-whisper base model.

This script is run after pip install when the audio-transcription extra
is enabled. It downloads the Whisper base model from Hugging Face
(Systran/faster-whisper-base) into the models/ directory so that audio
transcription works offline.
"""

import os
import sys
from pathlib import Path


def main():
    # Check if faster-whisper is installed (indicates audio-transcription extra)
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("[post-install] faster-whisper not installed, skipping whisper model download.")
        return 0

    # Determine the models directory relative to the package location.
    # src/common/post_install_model.py -> parent is src/common, parent.parent is src
    # The models directory is at the project root: {project_root}/models/
    script_dir = Path(__file__).resolve().parent
    # src/common -> src -> project_root
    project_root = script_dir.parent.parent
    model_dir = project_root / "models" / "models--Systran--faster-whisper-base"

    # Skip if model already exists
    if model_dir.exists() and (model_dir / "blobs").exists() and any(
        (model_dir / "blobs").iterdir()
    ):
        print(f"[post-install] Whisper model already exists at {model_dir}, skipping download.")
        return 0

    # Download using huggingface_hub (already a dependency of faster-whisper)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[post-install] huggingface_hub not available, cannot download whisper model.")
        return 1

    print(f"[post-install] Downloading whisper-base model from Hugging Face (Systran/faster-whisper-base)...")
    print(f"[post-install] Target directory: {model_dir}")

    try:
        snapshot_download(
            repo_id="Systran/faster-whisper-base",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[post-install] Model downloaded successfully to {model_dir}")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[post-install] Failed to download whisper model: {exc}", file=sys.stderr)
        print("[post-install] Audio transcription will attempt to download the model at runtime.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
