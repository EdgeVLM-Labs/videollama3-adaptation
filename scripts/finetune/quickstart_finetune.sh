#!/bin/bash

# VideoLLaMA3 QVED Finetuning - Quick Start
# This script runs all necessary steps to start finetuning

set -e  # Exit on error

# Setup logging
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/finetune_${TIMESTAMP}.log"
echo "Logging all output to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "VideoLLaMA3 QVED Finetuning - Quick Start"
echo "========================================="

# Step 1: Verify setup
echo -e "\n[Step 1/4] Verifying setup..."
bash scripts/finetune/verify_qved_setup.sh

# Step 2: Confirm to proceed
echo -e "\n========================================="
echo -n "Setup verified! Start finetuning? (y/N): "
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 3: Start finetuning
echo -e "\n[Step 2/4] Starting VideoLLaMA3 finetuning..."
echo "========================================="

# Run finetuning
bash scripts/finetune/finetune_qved.sh

# Generate training plots
echo ""
echo "Generating training plots..."
python utils/plot_training_stats.py \
  --log_file "$LOG_FILE" \
  --model_name "qved_finetune_videollama3_2b"

if [ $? -eq 0 ]; then
    echo "✓ Training plots generated successfully!"
    echo "  Location: plots/qved_finetune_videollama3_2b/"
else
    echo "⚠ Warning: Failed to generate plots. You can generate them later with:"
    echo "  python utils/plot_training_stats.py --log_file $LOG_FILE"
fi

echo -e "\n========================================="
echo "[Step 3/4] Finetuning complete!"
echo "========================================="
echo "Model saved to: results/qved_finetune/"
echo ""
echo "Finding latest checkpoint..."
LATEST_CKPT=$(ls -d results/qved_finetune/*/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "✓ Latest checkpoint: $LATEST_CKPT"
    MODEL_PATH="$LATEST_CKPT"
else
    echo "⚠ No checkpoints found. Model may still be in results/qved_finetune/"
    MODEL_PATH="results/qved_finetune"
fi

# Training output is logged to logs/ directory

# Step 4: Upload to HuggingFace
echo -e "\n========================================="
echo "[Step 4/4] Upload to HuggingFace"
echo "========================================="
echo -n "Upload finetuned model to HuggingFace? (y/N): "
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    HF_REPO_NAME="videollama3-qved-finetune-${TIMESTAMP}"
    echo "Uploading to EdgeVLM-Labs/${HF_REPO_NAME} (private repository)..."
    python utils/hf_upload.py \
        --model_path "$MODEL_PATH" \
        --repo_name "$HF_REPO_NAME" \
        --org "EdgeVLM-Labs"

    if [ $? -eq 0 ]; then
        echo "✓ Model uploaded successfully to HuggingFace!"
        echo "  URL: https://huggingface.co/EdgeVLM-Labs/${HF_REPO_NAME}"
    else
        echo "⚠ Warning: Failed to upload model to HuggingFace"
        echo "  You can upload manually later with:"
        echo "  python utils/hf_upload.py --model_path $MODEL_PATH"
    fi
else
    echo "Skipping HuggingFace upload."
    echo "You can upload later with:"
    echo "  python utils/hf_upload.py --model_path $MODEL_PATH"
fi

echo -e "\n========================================="
echo "All steps complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test the finetuned model:"
echo "   python inference/example_videollama3.py \\"
echo "     --model_path $MODEL_PATH \\"
echo "     --video_path dataset/squats/00255568.mp4"
echo ""
echo "2. Launch Gradio demo:"
echo "   python inference/launch_gradio_demo.py \\"
echo "     --model_path $MODEL_PATH"
echo ""
echo "3. Run test set evaluation:"
echo "   bash scripts/inference/run_inference.sh --model_path $MODEL_PATH"
echo ""
echo "4. Check model checkpoint details:"
echo "   ls -lh $MODEL_PATH/"
echo ""
echo "Training log saved to: $LOG_FILE"
echo "========================================="
