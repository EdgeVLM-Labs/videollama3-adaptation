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

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║         VideoLLaMA3 Model Downloader                       ║
║  Pre-downloads VideoLLaMA3-2B model for finetuning         ║
╚════════════════════════════════════════════════════════════╝
    """)

    model_id = "DAMO-NLP-SG/VideoLLaMA3-2B"
    
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

        print(f"\n{'='*60}")
        print("✓ Model downloaded successfully!")
        print(f"{'='*60}")
        print(f"  Cached at: {cache_dir}")
        print("\nYou can now start finetuning with:")
        print("  bash scripts/finetune/finetune_qved.sh")
        return 0

    except Exception as e:
        print(f"\n✗ Failed to download {model_id}")
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
                print(f"\n✓ Model downloaded via huggingface-cli!")
                print("\nYou can now start finetuning with:")
                print("  bash scripts/finetune/finetune_qved.sh")
                return 0
            else:
                print(f"✗ Fallback download failed: {result.stderr}")
        except Exception as fallback_error:
            print(f"✗ Fallback download failed: {fallback_error}")

        print("\n⚠ Download failed.")
        print("Please check your internet connection and try again.")
        print("\nManual download option:")
        print("  huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-2B")
        return 1

if __name__ == "__main__":
    sys.exit(main())
