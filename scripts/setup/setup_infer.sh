#!/bin/bash

echo "----------------------------------------------"
echo "Setting up Videollama3 inference environment"
echo "----------------------------------------------"
cd ../..
echo ""

echo "----------------------------------------------"
echo "Installing Miniconda..."
echo "----------------------------------------------"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH" #conda commands are found first
conda init bash

source $HOME/miniconda/etc/profile.d/conda.sh #initialize conda for bash

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "----------------------------------------------"
echo "Creating conda environment (videollama3-infer)..."
echo "----------------------------------------------"
conda create -n videollama3-infer python=3.11 -y
conda activate videollama3-infer

pip install --upgrade pip

cd VideoLLaMA3

echo ""
echo "----------------------------------------------"
echo "Installing PyTorch and dependencies..."
echo "----------------------------------------------"
pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

echo ""
echo "----------------------------------------------"
echo "Installing Flash Attention..."
echo "----------------------------------------------"

pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation --upgrade

echo ""
echo "----------------------------------------------"
echo "Installing Transformers and Accelerate..."
echo "----------------------------------------------"

pip install transformers==4.46.3 accelerate==1.0.1

echo ""
echo "----------------------------------------------"
echo "Installing video processing dependencies..."
echo "----------------------------------------------"

pip install decord ffmpeg-python imageio opencv-python
pip install hf_transfer

echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "❌ nvcc not found"
nvidia-smi 2>/dev/null || echo "❌ nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ PyTorch cannot see CUDA')
"

echo ""
echo "=== Flash Attention Check ==="
python -c "
try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('❌ Flash Attention not installed')
"

echo ""
echo "----------------------------------------------"
echo "Videollama3 inference environment setup complete!"
echo "----------------------------------------------"

# Initialize WandB
echo "🔑 Logging into WandB..."
wandb login

# Initialize HuggingFace Hub
echo "🤗 Logging into HuggingFace Hub..."
huggingface-cli login

source ~/.bashrc
