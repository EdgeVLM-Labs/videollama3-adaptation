#!/usr/bin/env python3
"""
Download required models for VideoLLaMA3 finetuning.
This ensures all models are cached before training starts.
Uses snapshot_download for efficient downloading without loading into memory.
"""

import os
import sys
from pathlib import Path

# Enable fast transfers
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def download_model(model_id, description):
    """Download a model from HuggingFace Hub using snapshot_download."""
    print(f"\n{'='*60}")
    print(f"Downloading {model_id}...")
    print(f"{'='*60}")

    try:
        from huggingface_hub import snapshot_download

        print("  → Downloading all model files...")
        cache_dir = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            resume_download=True,
            local_files_only=False,
        )

        print(f"✓ {model_id} downloaded successfully!")
        print(f"  Cached at: {cache_dir}")
        return True

    except Exception as e:
        print(f"✗ Failed to download {model_id}")
        print(f"  Error: {e}")

        # Try with huggingface-cli as fallback
        print("\n  Attempting fallback download with huggingface-cli...")
        try:
            import subprocess
            result = subprocess.run(
                ["huggingface-cli", "download", model_id, "--repo-type", "model"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✓ {model_id} downloaded via huggingface-cli!")
                return True
            else:
                print(f"✗ Fallback download failed: {result.stderr}")
        except Exception as fallback_error:
            print(f"✗ Fallback download failed: {fallback_error}")

        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║         VideoLLaMA3 Model Downloader                       ║
║  Pre-downloads all required models for finetuning          ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Models to download
    models = [
        ("DAMO-NLP-SG/VideoLLaMA3-2B", "Main VideoLLaMA3 model"),
        ("DAMO-NLP-SG/SigLIP-NaViT", "Vision encoder (SigLIP)"),
    ]

    results = []

    for model_id, description in models:
        print(f"\n[{len(results)+1}/{len(models)}] {description}")
        success = download_model(model_id, description)
        results.append((model_id, success))

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")

    for model_id, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {model_id}")

    all_success = all(success for _, success in results)

    if all_success:
        print("\n✓ All models downloaded successfully!")
        print("You can now start finetuning with:")
        print("  bash scripts/finetune/finetune_qved.sh")
        return 0
    else:
        print("\n⚠ Some models failed to download.")
        print("Please check your internet connection and try again.")
        print("\nManual download option:")
        print("  huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-2B")
        print("  huggingface-cli download DAMO-NLP-SG/SigLIP-NaViT")
        return 1

if __name__ == "__main__":
    sys.exit(main())
