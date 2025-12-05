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

echo ""
echo "----------------------------------------------"
echo "Installing Flash Attention..."
echo "----------------------------------------------"
# Flash-attn pinned to a compatible version
# pip install flash-attn==2.7.3 --no-build-isolation --upgrade
pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation --upgrade

echo ""
echo "----------------------------------------------"
echo "Installing Transformers and Accelerate..."
echo "----------------------------------------------"
# Transformers and accelerate
pip install transformers==4.46.3 accelerate==1.0.1

echo ""
echo "----------------------------------------------"
echo "Installing video processing dependencies..."
echo "----------------------------------------------"
# Video processing dependencies
pip install decord ffmpeg-python imageio opencv-python

 pip install hf_transfer

echo ""
echo "----------------------------------------------"
echo "Videollama3 inference environment setup complete!"
echo "----------------------------------------------"
