#!/bin/bash

# VideoLLaMA3 QVED Test Inference and Evaluation Report Generator
# This script runs inference on the test set and generates an evaluation report
# Example usage:
#   bash scripts/inference/run_inference.sh --model_path results/qved_finetune/qved-finetune-20251221/checkpoint-20

set -e  # Exit on error

echo "========================================="
echo "VideoLLaMA3 Test Inference & Evaluation"
echo "========================================="

# Default values
MODEL_PATH=""
TEST_JSON="dataset/qved_test.json"
DATA_PATH="dataset"
OUTPUT_DIR=""
DEVICE="cuda:0"
MAX_NEW_TOKENS=64
FPS=1
MAX_FRAMES=32
LIMIT=""
NO_BERT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test_json)
            TEST_JSON="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --max_frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --no-bert)
            NO_BERT="--no-bert"
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/inference/run_inference.sh --model_path <path> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path      Path to finetuned VideoLLaMA3 model checkpoint"
            echo ""
            echo "Optional:"
            echo "  --test_json       Path to test set JSON (default: dataset/qved_test.json)"
            echo "  --data_path       Base path for video files (default: dataset)"
            echo "  --output_dir      Output directory for results (default: model directory)"
            echo "  --device          Device to use: cuda/cpu (default: cuda)"
            echo "  --max_new_tokens  Max tokens to generate (default: 512)"
            echo "  --fps             Video FPS for processing (default: 1)"
            echo "  --max_frames      Max frames to extract from video (default: 32)"
            echo "  --limit           Limit number of samples (for testing)"
            echo "  --no-bert         Skip BERT similarity (faster evaluation)"
            echo ""
            echo "Examples:"
            echo "  # Using local checkpoint:"
            echo "  bash scripts/inference/run_inference.sh --model_path results/qved_finetune/qved-finetune-20251221/checkpoint-20"
            echo ""
            echo "  # With options:"
            echo "  bash scripts/inference/run_inference.sh --model_path results/qved_finetune/run1 --limit 10 --max_frames 64"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate: model_path must be provided
if [ -z "$MODEL_PATH" ]; then
    echo "❌ Error: --model_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate model path exists
if [ ! -e "$MODEL_PATH" ]; then
    echo "❌ Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Auto-detect latest checkpoint if path is a directory
if [ -d "$MODEL_PATH" ]; then
    # Check if this is already a checkpoint directory
    if [[ "$MODEL_PATH" != *"checkpoint-"* ]]; then
        # Look for checkpoint subdirectories
        LATEST_CKPT=$(ls -d "$MODEL_PATH"/*/checkpoint-* 2>/dev/null | sort -V | tail -1)

        if [ -z "$LATEST_CKPT" ]; then
            # Try direct checkpoint-* in current directory
            LATEST_CKPT=$(ls -d "$MODEL_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        fi

        if [ -n "$LATEST_CKPT" ]; then
            echo "📍 Auto-detected latest checkpoint: $LATEST_CKPT"
            MODEL_PATH="$LATEST_CKPT"
        else
            echo "ℹ️  No checkpoints found, using base model directory: $MODEL_PATH"
        fi
    fi
fi

if [ ! -f "$TEST_JSON" ]; then
    echo "❌ Error: Test JSON not found: $TEST_JSON"
    exit 1
fi

# Set output directory if not already set
if [ -z "$OUTPUT_DIR" ]; then
    # Extract base directory (remove checkpoint-XX if present)
    if [[ "$MODEL_PATH" == *"checkpoint-"* ]]; then
        OUTPUT_DIR=$(dirname "$MODEL_PATH")
    else
        OUTPUT_DIR="$MODEL_PATH"
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

PREDICTIONS_FILE="${OUTPUT_DIR}/test_predictions.json"
REPORT_FILE="${OUTPUT_DIR}/test_evaluation_report.xlsx"

echo ""
echo "Configuration:"
echo "  Model path:      $MODEL_PATH"
echo "  Test JSON:       $TEST_JSON"
echo "  Data path:       $DATA_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Device:          $DEVICE"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Video FPS:       $FPS"
echo "  Max frames:      $MAX_FRAMES"
if [ -n "$LIMIT" ]; then
    echo "  Sample limit:    $LIMIT"
fi
echo "========================================="
echo ""

# Step 1: Run inference
echo "[Step 1/2] Running inference on test set..."
echo "========================================="

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

python utils/test_inference.py \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --data_path "$DATA_PATH" \
    --output "$PREDICTIONS_FILE" \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --fps "$FPS" \
    --max_frames "$MAX_FRAMES" \
    $LIMIT_ARG

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi

echo ""
echo "✓ Inference complete! Predictions saved to: $PREDICTIONS_FILE"
echo ""

# Step 2: Generate evaluation report
echo "[Step 2/2] Generating evaluation report..."
echo "========================================="

python utils/generate_test_report.py \
    --predictions "$PREDICTIONS_FILE" \
    --output "$REPORT_FILE" \
    $NO_BERT

if [ $? -ne 0 ]; then
    echo "⚠ Warning: Failed to generate evaluation report"
    echo "  You can generate it later with:"
    echo "  python utils/generate_test_report.py --predictions $PREDICTIONS_FILE"
else
    echo ""
    echo "✓ Evaluation report saved to: $REPORT_FILE"
fi

echo ""
echo "========================================="
echo "✅ All steps complete!"
echo "========================================="
echo ""
echo "Output files:"
echo "  Predictions: $PREDICTIONS_FILE"
echo "  Report:      $REPORT_FILE"
echo ""
echo "========================================="
