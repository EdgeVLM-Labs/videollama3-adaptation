#!/bin/bash

echo "----------------------------------------------"
echo "Setting up VideoLLaMA3 training environment"
echo "----------------------------------------------"
cd ..
echo ""

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH" #conda commands are found first
conda init bash

source $HOME/miniconda/etc/profile.d/conda.sh #initialize conda for bash

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "----------------------------------------------"
echo "Creating conda environment (videollama3-train)..."
echo "----------------------------------------------"
conda create -n videollama3-train python=3.11
conda activate videollama3-train

pip install --upgrade pip

cd videollama3-adaptation

echo ""
echo "----------------------------------------------"
echo "Installing requirements..."
echo "----------------------------------------------"
pip install -r requirements.txt

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

echo ""
echo "----------------------------------------------"
echo "Installing Flash Attention..."
echo "----------------------------------------------"

pip install ninja packaging wheel huggingface_hub hf_transfer huggingface_hub[cli]
pip install "flash-attn==2.7.3" --no-build-isolation
pip install openpyxl scikit-learn sentence-transformers rouge_score scikit-image jsonb

conda install -n videollama3-train -c conda-forge ffmpeg -y

apt-get update
apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

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
echo "VideoLLaMA3 training environment setup complete!"
echo "----------------------------------------------"

# Initialize WandB
echo "🔑 Logging into WandB..."
wandb login

# Initialize HuggingFace Hub
echo "🤗 Logging into HuggingFace Hub..."
huggingface-cli login

source /root/.bashrc
