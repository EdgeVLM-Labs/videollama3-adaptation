#!/bin/bash

echo "----------------------------------------------"
echo "Starting VideoLLaMA3 Fine-tuning"
echo "----------------------------------------------"

# Set HuggingFace environment variables for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

# PyTorch memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-download models if not cached
echo ""
echo "----------------------------------------------"
echo "Checking for required models..."
echo "----------------------------------------------"

# Check if models exist in cache
MODELS_EXIST=true
HF_CACHE="${HF_HOME}/hub"

if [ ! -d "${HF_CACHE}/models--DAMO-NLP-SG--VideoLLaMA3-2B" ]; then
    echo "✗ VideoLLaMA3-2B not found in cache"
    MODELS_EXIST=false
fi

if [ ! -d "${HF_CACHE}/models--DAMO-NLP-SG--SigLIP-NaViT" ]; then
    echo "✗ SigLIP-NaViT not found in cache"
    MODELS_EXIST=false
fi

if [ "$MODELS_EXIST" = false ]; then
    echo ""
    echo "Downloading required models..."
    python3 scripts/finetune/download_models.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠ Warning: Model download encountered issues."
        echo "Continuing anyway - training script will attempt download..."
        echo ""
    fi
else
    echo "✓ All required models found in cache"
fi

echo ""
echo "----------------------------------------------"
echo "Models ready. Starting training..."
echo "----------------------------------------------"

# Ensure we're in the project root
# When running: bash scripts/finetune/quickstart_finetune.sh from project root
echo "Current working directory: $(pwd)"

# Verify dataset accessibility
if [ ! -d "dataset" ]; then
    echo "✗ ERROR: Dataset directory not found"
    echo "  Expected: $(pwd)/dataset"
    echo "  Please run from project root: /workspace/videollama3-adaptation"
    exit 1
fi

echo "✓ Dataset directory found at: $(pwd)/dataset"
VIDEO_COUNT=$(find dataset -name "*.mp4" -type f 2>/dev/null | wc -l)
echo "✓ Found $VIDEO_COUNT video files"

# Verify training data file
if [ ! -f "dataset/qved_train.json" ]; then
    echo "✗ ERROR: Training data not found at dataset/qved_train.json"
    exit 1
fi
echo "✓ Training data found: dataset/qved_train.json"

echo ""

echo "----------------------------------------------"
echo "Starting Stage 4 Fine-tuning..."
echo "----------------------------------------------"
echo ""

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-1}  # Single GPU setup
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=4
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

echo "Training configuration:"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Local Batch Size: $LOCAL_BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size: $(($LOCAL_BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"

# Log Arguments
export WANDB_PROJECT="videollama3"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="qved-finetune-$(date +%Y%m%d_%H%M%S)"

# Model checkpoint - use HuggingFace model or local checkpoint
# Option 1: HuggingFace model (recommended for initial fine-tuning)
MODEL_PATH="DAMO-NLP-SG/VideoLLaMA3-2B"
# Option 2: Local checkpoint
# MODEL_PATH="work_dirs/videollama3/stage_3/checkpoint-xxxx"

PRECEDING_RUN_NAME=stage_4
RUN_NAME=stage_4
DATA_DIR=dataset
OUTP_DIR=work_dirs/videollama3

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama3/train.py \
    --deepspeed scripts/zero3.json \
    --model_type videollama3_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path ${DATA_DIR}/qved_train.json \
    --data_folder ${DATA_DIR} \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 32 \
    --model_max_length 2048 \
    --mm_max_length 2048 \
    --use_token_compression True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --llm_lr 2e-5 \
    --mm_projector_lr 1e-5 \
    --vision_encoder_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name $RUN_NAME

echo ""
echo "----------------------------------------------"
echo "Fine-tuning completed!"
echo "----------------------------------------------"
