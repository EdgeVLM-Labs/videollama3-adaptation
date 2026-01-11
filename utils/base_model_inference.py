#!/usr/bin/env python3
"""
Base Model Inference Script for QVED Test Set.

Runs inference using a base VideoLLaMA3 model on the test JSON file and
stores predictions to an output JSON.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def load_base_model(model_path: str, device: str = "cuda:0"):
    """Load the base VideoLLaMA3 model and processor."""
    if not model_path:
        model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"

    print(f"Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else None,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    return model, processor


@torch.inference_mode()
def run_inference(
    model,
    processor,
    video_path: str,
    prompt: str,
    fps: int = 1,
    max_frames: int = 16,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
):
    """Run inference on a single video and return the decoded response."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": fps,
                        "max_frames": max_frames,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs and device.startswith("cuda"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    start_time = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    generation_time = time.time() - start_time

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    metrics = {
        "generation_time": round(generation_time, 4),
        "generated_tokens": int(output_ids.shape[1] - inputs["input_ids"].shape[1]),
    }

    return response, metrics


def main():
    parser = argparse.ArgumentParser(description="Run base model inference on test set")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--base_model", type=str, default="",
                        help="Base model path or HF ID")
    parser.add_argument("--data_path", type=str, default="dataset", help="Base path for videos")
    parser.add_argument("--output", type=str, default="base_model_predictions.json",
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device: cuda/cpu")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")

    args = parser.parse_args()

    model, processor = load_base_model(args.base_model, device=args.device)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")

    results = []
    print(f"Running inference on {len(test_data)} samples...")

    for item in tqdm(test_data, desc="Processing videos"):
        video_field = item.get("video")
        video_rel_path = video_field[0] if isinstance(video_field, list) else video_field
        video_path = str(Path(args.data_path) / video_rel_path)

        conversations = item.get("conversations", [])
        prompt = conversations[0]["value"].replace("<video>", "").strip() if conversations else ""
        ground_truth = conversations[1]["value"] if len(conversations) > 1 else ""

        try:
            prediction, metrics = run_inference(
                model,
                processor,
                video_path,
                prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )

            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "generated_tokens": metrics["generated_tokens"],
                "generation_time": metrics["generation_time"],
                "status": "success",
            })
        except Exception as exc:
            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": "",
                "status": "error",
                "error": str(exc),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
