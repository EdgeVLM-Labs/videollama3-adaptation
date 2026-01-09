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

# Verify validation data file
if [ ! -f "dataset/qved_val.json" ]; then
    echo "✗ ERROR: Validation data not found at dataset/qved_val.json"
    exit 1
fi
echo "✓ Validation data found: dataset/qved_val.json"

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

# ============================================
# Training Hyperparameters
# ============================================

# Video Processing
FPS=1                        # Frame sampling rate
MAX_FRAMES=16                # Maximum frames per video

# Learning Rates
LR=2e-4                      # LLM learning rate
MM_PROJ_LR=1e-4              # Projector learning rate

# LoRA Configuration
LORA_R=64                    # LoRA rank
LORA_ALPHA=128               # LoRA alpha

# Batch & Training
BATCH_SIZE=8                 # Per-device train batch size
EVAL_BATCH_SIZE=8            # Per-device eval batch size
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=3

# Sequence Length
MAXLEN=2048                  # Max sequence length

# Evaluation & Saving
EVAL_STRATEGY="steps"
EVAL_STEPS=30
SAVE_STRATEGY="steps"
SAVE_STEPS=30

# Scheduler & Optimization
WARMUP_RATIO=0.05

# Misc Training
GRADIENT_CHECKPOINTING=True
LOGGING_STEPS=1
DATALOADER_NUM_WORKERS=2

echo ""
echo "============================================"
echo "       Training Hyperparameters"
echo "============================================"
echo ""
echo "Video Processing:"
echo "  Frame Sampling Rate (FPS):    $FPS"
echo "  Maximum Frames per Video:     $MAX_FRAMES"
echo ""
echo "Learning Rates:"
echo "  LLM Learning Rate:            $LR"
echo "  Projector Learning Rate:      $MM_PROJ_LR"
echo ""
echo "LoRA Configuration:"
echo "  LoRA Rank (r):                $LORA_R"
echo "  LoRA Alpha:                   $LORA_ALPHA"
echo ""
echo "Batch & Training:"
echo "  Epochs:                       $NUM_EPOCHS"
echo "  Batch Size (per device):      $BATCH_SIZE"
echo "  Gradient Accumulation Steps:  $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size:         $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"
echo ""
echo "Sequence Length:"
echo "  Max Sequence Length:          $MAXLEN"
echo ""
echo "Evaluation & Saving:"
echo "  Eval Strategy:                $EVAL_STRATEGY"
echo "  Eval Steps:                   $EVAL_STEPS"
echo "  Save Strategy:                $SAVE_STRATEGY"
echo "  Save Steps:                   $SAVE_STEPS"
echo ""
echo "Scheduler & Optimization:"
echo "  Warmup Ratio:                 $WARMUP_RATIO"
echo ""
echo "Misc:"
echo "  Gradient Checkpointing:       $GRADIENT_CHECKPOINTING"
echo "  Logging Steps:                $LOGGING_STEPS"
echo "  Dataloader Workers:           $DATALOADER_NUM_WORKERS"
echo ""
echo "============================================"
echo ""

# Log Arguments
export WANDB_PROJECT="videollama3"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="qved-finetune-$(date +%Y%m%d_%H%M%S)"

RUN_NAME=$WANDB_NAME
DATA_DIR=dataset
OUTP_DIR=results/qved_finetune

# Save hyperparameters to a config file
mkdir -p "$OUTP_DIR"
CONFIG_FILE="$OUTP_DIR/hyperparameters.json"
cat <<EOF > "$CONFIG_FILE"
{
  "base_model": "$MODEL_PATH",
  "dataset": "QVED",
  "frame_sampling_rate": $FPS,
  "max_frames_per_video": $MAX_FRAMES,
  "epochs": $NUM_EPOCHS,
  "learning_rate": "$LR",
  "projector_lr": "$MM_PROJ_LR",
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "batch_size": $BATCH_SIZE,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "max_sequence_length": $MAXLEN,
  "gradient_checkpointing": $GRADIENT_CHECKPOINTING,
  "per_device_eval_batch_size": $EVAL_BATCH_SIZE,
  "eval_strategy": "$EVAL_STRATEGY",
  "eval_steps": $EVAL_STEPS,
  "save_strategy": "$SAVE_STRATEGY",
  "save_steps": $SAVE_STEPS,
  "warmup_ratio": $WARMUP_RATIO,
  "logging_steps": $LOGGING_STEPS,
  "dataloader_num_workers": $DATALOADER_NUM_WORKERS,
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
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --lora_bias none \
    --model_type videollama3_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_encoder "$VISION_TOWER" \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --data_path ${DATA_DIR}/qved_train.json \
    --eval_data_path ${DATA_DIR}/qved_val.json \
    --data_folder ${DATA_DIR} \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps $FPS \
    --max_frames $MAX_FRAMES \
    --model_max_length $MAXLEN \
    --mm_max_length 1536 \
    --use_token_compression True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${RUN_NAME} \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --llm_lr $LR \
    --mm_projector_lr $MM_PROJ_LR \
    --vision_encoder_lr None \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps $LOGGING_STEPS \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --report_to wandb \
    --run_name $RUN_NAME

echo ""
echo "----------------------------------------------"
echo "Fine-tuning completed!"
echo "----------------------------------------------"
