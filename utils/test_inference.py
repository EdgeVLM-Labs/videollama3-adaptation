#!/usr/bin/env python3
"""
VideoLLaMA3 Test Inference Script for QVED Dataset

This script runs inference on videos from the QVED test set using a finetuned VideoLLaMA3 model.
It loads videos from qved_test.json and generates predictions.

Usage:
    python utils/test_inference.py --model_path results/qved_finetune/run1/checkpoint-20
    python utils/test_inference.py --model_path results/qved_finetune/run1 --output test_predictions.json
"""

import sys
import os
import warnings
import argparse
import jsonb
import time
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")


def load_model(model_path: str, device: str = "cuda:0"):
    """
    Load the finetuned VideoLLaMA3 model and processor.

    Args:
        model_path: Path to finetuned model checkpoint
        device: Device to load model on

    Returns:
        tuple: (model, processor)
    """
    print(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # load from base model
    base_model = "DAMO-NLP-SG/VideoLLaMA3-2B"
    print(f"Loading processor from base model: {base_model}")
    processor = AutoProcessor.from_pretrained(
        base_model,
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
    max_frames: int = 32,
    max_new_tokens: int = 512,
    device: str = "cuda"
):
    """
    Run inference on a single video.

    Args:
        model: VideoLLaMA3 model
        processor: VideoLLaMA3 processor
        video_path: Path to video file
        prompt: Text prompt for the model
        fps: Frames per second to extract
        max_frames: Maximum number of frames to use
        max_new_tokens: Maximum tokens to generate
        device: Device to use

    Returns:
        tuple: (prediction_text, metrics_dict)
    """
    # Build conversation
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
                        "max_frames": max_frames
                    }
                },
                {"type": "text", "text": prompt},
            ]
        },
    ]

    # Process inputs
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # Track metrics
    input_token_count = inputs['input_ids'].shape[1]
    start_time = time.time()

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding
        use_cache=True
    )

    end_time = time.time()
    generation_time = end_time - start_time

    # Decode output
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Calculate metrics
    output_token_count = output_ids.shape[1]
    generated_token_count = output_token_count - input_token_count
    tokens_per_second = generated_token_count / generation_time if generation_time > 0 else 0

    metrics = {
        'generated_tokens': generated_token_count,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second
    }

    return response, metrics


def warmup_gpu(model, processor, warmup_videos: list, **kwargs):
    """Warm up GPU with sample videos before actual inference."""
    print("\n🔥 Warming up GPU...")
    for video_path in warmup_videos[:2]:  # Use 2 videos for warmup
        if not os.path.exists(video_path):
            continue
        try:
            prompt = "Please evaluate this exercise form."
            _ = run_inference(model, processor, video_path, prompt, **kwargs)
        except Exception as e:
            print(f"  ⚠ Warmup warning for {video_path}: {e}")
    print("✓ GPU warmup complete")


def main():
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 inference on QVED test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned VideoLLaMA3 model checkpoint")
    parser.add_argument("--test_json", type=str, default="dataset/qved_test.json",
                        help="Path to test set JSON")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Base path for video files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions (default: saves to model directory)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for video processing")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="Maximum frames to extract from video")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")

    args = parser.parse_args()

    # Set default output path to model directory if not provided
    if args.output is None:
        # Extract base directory (remove checkpoint-XX if present)
        if "checkpoint-" in args.model_path:
            base_dir = str(Path(args.model_path).parent)
        else:
            base_dir = args.model_path
        args.output = str(Path(base_dir) / "test_predictions.json")
        print(f"Output will be saved to: {args.output}")

    # Load model
    print(f"\n📦 Loading VideoLLaMA3 model...")
    model, processor = load_model(args.model_path, device=args.device)
    print("✓ Model loaded successfully")

    # Load test data
    print(f"\n📋 Loading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Total test samples: {len(test_data)}")

    # GPU warmup with first few test videos
    if args.device == "cuda:0" and len(test_data) > 0:
        warmup_videos = [
            str(Path(args.data_path) / item['video'][0] if isinstance(item['video'], list) else item['video'])
            for item in test_data[:2]
        ]
        warmup_gpu(
            model, processor, warmup_videos,
            fps=args.fps,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            device=args.device
        )

    # Run inference
    results = []
    throughput_stats = []
    print("\n🎬 Running inference...")

    for item in tqdm(test_data, desc="Processing videos"):
        # Handle video path (could be list or string)
        video_field = item['video']
        if isinstance(video_field, list):
            video_rel_path = video_field[0]
        else:
            video_rel_path = video_field

        video_path = str(Path(args.data_path) / video_rel_path)

        # Extract prompt and ground truth
        conversations = item['conversations']

        # Remove <video> tag from prompt if present (processor handles it)
        prompt = conversations[0]['value'].replace('<video>', '').strip()
        ground_truth = conversations[1]['value']

        try:
            # Run inference
            prediction, metrics = run_inference(
                model, processor,
                video_path, prompt,
                fps=args.fps,
                max_frames=args.max_frames,
                max_new_tokens=args.max_new_tokens,
                device=args.device
            )

            throughput_stats.append(metrics['tokens_per_second'])

            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "generated_tokens": metrics['generated_tokens'],
                "generation_time": round(metrics['generation_time'], 4),
                "tokens_per_second": round(metrics['tokens_per_second'], 2),
                "status": "success"
            })

        except Exception as e:
            print(f"\n✗ Error processing {video_rel_path}: {str(e)}")
            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": "",
                "status": "error",
                "error": str(e)
            })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    # Calculate throughput statistics
    if throughput_stats:
        avg_throughput = np.mean(throughput_stats)
        median_throughput = np.median(throughput_stats)
        min_throughput = np.min(throughput_stats)
        max_throughput = np.max(throughput_stats)

    print(f"\n{'='*60}")
    print("✅ Inference Complete!")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if throughput_stats:
        print(f"\n📊 Throughput Statistics (Tokens/Second):")
        print(f"  Mean:   {avg_throughput:.2f}")
        print(f"  Median: {median_throughput:.2f}")
        print(f"  Min:    {min_throughput:.2f}")
        print(f"  Max:    {max_throughput:.2f}")

    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
