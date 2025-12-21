#!/bin/bash

echo "----------------------------------------------"
echo "Starting VideoLLaMA3 Fine-tuning"
echo "----------------------------------------------"

# Set HuggingFace environment variables for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

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

# # Initialize conda
# source $HOME/miniconda/etc/profile.d/conda.sh

# # Activate the training environment
# conda activate videollama3-train

# if [ $? -ne 0 ]; then
#     echo "Error: Failed to activate conda environment 'videollama3-train'"
#     echo "Please ensure the environment is created using setup_train.sh"
#     exit 1
# fi

# echo "Environment activated successfully!"
# echo ""

# # Navigate to the project root
# cd "$(dirname "$0")/../.."

echo "----------------------------------------------"
echo "Starting Stage 4 Fine-tuning..."
echo "----------------------------------------------"
echo ""

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
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
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export WANDB_PROJECT="videollama3"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="qved-finetune-$(date +%Y%m%d_%H%M%S)"

# Model checkpoint - use HuggingFace model or local checkpoint
# Option 1: HuggingFace model (recommended for initial fine-tuning)
MODEL_PATH="DAMO-NLP-SG/VideoLLaMA3-2B"
# Option 2: Local checkpoint (if you have one from previous training)
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
    --deepspeed scripts/zero2.json \
    --model_type videollama3_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path ${DATA_DIR}/qved_train.json \
    --data_folder ${DATA_DIR} \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 180 \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --use_token_compression True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --llm_lr 2e-5 \
    --mm_projector_lr 1e-5 \
    --vision_encoder_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --dataset_cache_dir /mnt/damovl/DAMOVL_DATASETS/.cache

echo ""
echo "----------------------------------------------"
echo "Fine-tuning completed!"
echo "----------------------------------------------"
