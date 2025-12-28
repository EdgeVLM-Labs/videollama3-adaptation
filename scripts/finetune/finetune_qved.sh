#!/bin/bash

echo "----------------------------------------------"
echo "Starting VideoLLaMA3 Fine-tuning"
echo "----------------------------------------------"

# Set HuggingFace environment variables for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

export PYTHONPATH="./:$PYTHONPATH"
# Suppress DeepSpeed hostfile warning for single-GPU training
export PDSH_RCMD_TYPE=ssh

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
    python3 utils/download_models.py

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

# Model checkpoint - use HuggingFace model or local checkpoint
# Option 1: HuggingFace model (recommended for initial fine-tuning)
MODEL_PATH="DAMO-NLP-SG/VideoLLaMA3-2B"
# Option 2: Local checkpoint
# MODEL_PATH="work_dirs/videollama3/stage_3/checkpoint-xxxx"
VISION_TOWER="DAMO-NLP-SG/SigLIP-NaViT"
PROJECTOR_TYPE="mlp2x_gelu"

# Training Arguments
LR=2e-4
MM_PROJ_LR=1e-4
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
NUM_EPOCHS=3
LORA_R=64                    # LoRA rank
LORA_ALPHA=128               # LoRA alpha
MAXLEN=2048                  # Max sequence length

echo "Training configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size: $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo "  Number of Epochs: $NUM_EPOCHS"

# Log Arguments
export WANDB_PROJECT="videollama3"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="qved-finetune-$(date +%Y%m%d_%H%M%S)"

RUN_NAME=$WANDB_NAME
DATA_DIR=dataset
OUTP_DIR=results/qved_finetune

# Save hyperparameters to a config file
CONFIG_FILE="$OUTP_DIR/hyperparameters.json"
cat <<EOF > "$CONFIG_FILE"
{
  "base_model": "$MODEL_PATH",
  "dataset": "QVED",
  "epochs": $EPOCHS,
  "learning_rate": $LR,
  "mm_projector_lr": $MM_PROJ_LR,
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "batch_size": $BATCH_SIZE,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "max_length": $MAXLEN,
  "wandb_project": "$WANDB_PROJECT",
  "wandb_entity": "$WANDB_ENTITY",
  "wandb_run_name": "$WANDB_NAME"
}
EOF
echo "Hyperparameters saved to $CONFIG_FILE"

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama3/train.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --model_type videollama3_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_encoder "$VISION_TOWER" \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --data_path ${DATA_DIR}/qved_train.json \
    --data_folder ${DATA_DIR} \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 32 \
    --model_max_length $MAXLEN  \
    --mm_max_length 1536 \
    --use_token_compression True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${RUN_NAME} \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 3 \
    --llm_lr $LR \
    --mm_projector_lr $MM_PROJ_LR \
    --vision_encoder_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to wandb \
    --run_name $RUN_NAME

echo ""
echo "----------------------------------------------"
echo "Fine-tuning completed!"
echo "----------------------------------------------"
