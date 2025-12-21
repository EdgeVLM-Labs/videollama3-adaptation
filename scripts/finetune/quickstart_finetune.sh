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
echo -e "\n[Step 1/3] Verifying setup..."
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
echo -e "\n[Step 2/3] Starting VideoLLaMA3 finetuning..."
echo "========================================="

# Run finetuning
bash scripts/finetune/finetune_qved.sh

echo -e "\n========================================="
echo "[Step 3/3] Finetuning complete!"
echo "========================================="
echo "Model saved to: work_dirs/videollama3/stage_4/"
echo ""
echo "Finding latest checkpoint..."
LATEST_CKPT=$(ls -d work_dirs/videollama3/stage_4/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "✓ Latest checkpoint: $LATEST_CKPT"
    MODEL_PATH="$LATEST_CKPT"
else
    echo "⚠ No checkpoints found. Model may still be in work_dirs/videollama3/stage_4/"
    MODEL_PATH="work_dirs/videollama3/stage_4"
fi

# Optional: Check tensorboard logs
echo -e "\n========================================="
echo "Training Logs and Monitoring"
echo "========================================="
if [ -d "work_dirs/videollama3/stage_4/runs" ]; then
    echo "✓ TensorBoard logs available"
    echo "  View with: tensorboard --logdir work_dirs/videollama3/stage_4/runs"
else
    echo "⚠ No TensorBoard logs found yet"
fi

echo -e "\n========================================="
echo "All steps complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. View training progress:"
echo "   tensorboard --logdir work_dirs/videollama3/stage_4/runs"
echo ""
echo "2. Test the finetuned model:"
echo "   python inference/example_videollama3.py \\"
echo "     --model_path $MODEL_PATH \\"
echo "     --video_path dataset/squats/00255568.mp4"
echo ""
echo "3. Launch Gradio demo:"
echo "   python inference/launch_gradio_demo.py \\"
echo "     --model_path $MODEL_PATH"
echo ""
echo "4. Check model checkpoint details:"
echo "   ls -lh $MODEL_PATH/"
echo ""
echo "Training log saved to: $LOG_FILE"
echo "========================================="
