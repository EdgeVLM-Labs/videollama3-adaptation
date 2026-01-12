#!/usr/bin/env python3
"""
HuggingFace Model Upload Utility

Uploads finetuned VideoLLaMA3 models to HuggingFace Hub.

Usage:
    python utils/hf_upload.py --model_path results/qved_finetune_videollama3_2B
    python utils/hf_upload.py --model_path results/qved_finetune_videollama3_2B --repo_name qved-finetune-20241128
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, login


# Default organization name
DEFAULT_ORG = "EdgeVLM-Labs"
DEFAULT_BASE_MODEL = "DAMO-NLP-SG/VideoLLaMA3-2B"


def get_default_repo_name() -> str:
    """Generate a default repository name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"qved-finetune-{timestamp}"


def check_hf_login() -> bool:
    """Check if user is logged into HuggingFace."""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception:
        return False


def upload_model_to_hf(
    model_path: str,
    repo_name: str = None,
    org_name: str = DEFAULT_ORG,
    private: bool = True,
    commit_message: str = None,
    model_card_path: str = None,
) -> str:
    """
    Upload a finetuned VideoLLaMA3 model to HuggingFace Hub.

    Args:
        model_path: Path to the model directory (can be checkpoint or base finetuning dir)
        repo_name: Name for the HuggingFace repository
        org_name: HuggingFace organization name
        private: Whether to create a private repository
        commit_message: Custom commit message

    Returns:
        URL of the uploaded model on HuggingFace
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Check for adapter files or model files
    has_adapter = (model_path / "adapter_config.json").exists()
    has_model = (model_path / "config.json").exists() or (model_path / "pytorch_model.bin").exists()

    if not has_adapter and not has_model:
        # Maybe it's a checkpoint directory
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            # Use the latest checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
            print(f"Using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
            has_adapter = (model_path / "adapter_config.json").exists()
            has_model = (model_path / "config.json").exists()

    if not has_adapter and not has_model:
        raise ValueError(
            f"No model or adapter files found in {model_path}. "
            "Expected adapter_config.json or config.json"
        )

    # Generate repo name if not provided
    if repo_name is None:
        repo_name = get_default_repo_name()

    # Full repository ID
    repo_id = f"{org_name}/{repo_name}"

    print(f"\n{'='*60}")
    print("HuggingFace Model Upload (VideoLLaMA3)")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print(f"Type: {'LoRA Adapter' if has_adapter else 'Full Model'}")
    print(f"{'='*60}\n")

    # Check login status
    if not check_hf_login():
        print("⚠ Not logged into HuggingFace. Please login first:")
        print("  huggingface-cli login")
        print("  or set HF_TOKEN environment variable")

        # Try to login with token from environment
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("\nFound HF_TOKEN in environment, attempting login...")
            login(token=hf_token)
        else:
            sys.exit(1)

    api = HfApi()

    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠ Warning: Could not create repository: {e}")
        print("  Will try to upload anyway...")

    # Prepare commit message
    if commit_message is None:
        if has_adapter:
            commit_message = f"Upload LoRA adapters from {model_path.name} (VideoLLaMA3)"
        else:
            commit_message = f"Upload finetuned VideoLLaMA3 model from {model_path.name}"

    # Ensure a model card exists
    ensure_model_card(
        model_path=model_path,
        repo_id=repo_id,
        has_adapter=has_adapter,
        model_card_path=model_card_path,
    )

    # Upload model
    print(f"\n🚀 Uploading model to {repo_id}...")
    print("  This may take a few minutes depending on model size...")

    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.py", "__pycache__", "*.pyc", "runs/*", "wandb/*"],
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

    # Get repository URL
    repo_url = f"https://huggingface.co/{repo_id}"

    print(f"\n{'='*60}")
    print("✅ Upload Complete!")
    print(f"{'='*60}")
    print(f"Repository URL: {repo_url}")
    print(f"\nTo use this model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
    if has_adapter:
        print(f"\n  # For LoRA adapters:")
        print(f"  from peft import PeftModel")
        print(f"  base_model = AutoModelForCausalLM.from_pretrained('DAMO-NLP-SG/VideoLLaMA3-2B')")
        print(f"  model = PeftModel.from_pretrained(base_model, '{repo_id}')")
    print(f"{'='*60}")

    return repo_url


def ensure_model_card(
    model_path: Path,
    repo_id: str,
    has_adapter: bool,
    model_card_path: str = None,
) -> None:
    """Create or copy a model card (README.md) into the model folder."""
    readme_path = model_path / "README.md"

    if model_card_path:
        source_path = Path(model_card_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model card not found: {source_path}")
        shutil.copyfile(source_path, readme_path)
        print(f"✓ Model card copied to: {readme_path}")
        return

    if readme_path.exists():
        print(f"✓ Existing model card found at: {readme_path}")
        return

    card = f"""---
language: en
license: apache-2.0
tags:
- videollama3
- video
- vision-language
- qved
base_model: {DEFAULT_BASE_MODEL}
library_name: transformers
---

# {repo_id}

## Model Description
This repository contains a fine-tuned VideoLLaMA3 model for video-based exercise form evaluation.
It is fine-tuned from `{DEFAULT_BASE_MODEL}` on the QVED dataset.

## Intended Use
- Analyze exercise videos and provide feedback on form and corrections.
- Research and evaluation for video-language tasks.

## Limitations
- Predictions may be incorrect or incomplete.
- Performance depends on video quality and framing.

## Training Data
QVED fine-grained labels converted into the VideoLLaMA3 conversation format.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("{DEFAULT_BASE_MODEL}", trust_remote_code=True)
```

## Citation
If you use this model, please cite:

```bibtex
@misc{{videollama3-qved-finetune,
  author = {{EdgeVLM Labs}},
  title = {{VideoLLaMA3 QVED Finetuned Model}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""

    readme_path.write_text(card, encoding="utf-8")
    print(f"✓ Model card created at: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload finetuned VideoLLaMA3 model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned model directory",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help=f"Name for the HuggingFace repository (default: qved-finetune-TIMESTAMP)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=DEFAULT_ORG,
        help=f"HuggingFace organization name (default: {DEFAULT_ORG})",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository (default: private)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default=None,
        help="Path to a README.md to upload as the model card",
    )

    args = parser.parse_args()

    # By default, repositories are private unless --public is specified
    is_private = not args.public

    try:
        repo_url = upload_model_to_hf(
            model_path=args.model_path,
            repo_name=args.repo_name,
            org_name=args.org,
            private=is_private,
            commit_message=args.commit_message,
            model_card_path=args.model_card,
        )
        print(f"\n🎉 Success! Model available at: {repo_url}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
