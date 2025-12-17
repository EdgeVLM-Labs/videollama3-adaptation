#!/bin/bash

echo "----------------------------------------------"
echo "Setting up VideoLLaMA3 training environment"
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
echo "Creating conda environment (videollama3-train)..."
echo "----------------------------------------------"
conda create -n videollama3-train python=3.11 -y
conda activate videollama3-train

pip install --upgrade pip

cd VideoLLaMA3

echo ""
echo "----------------------------------------------"
echo "Installing requirements..."
echo "----------------------------------------------"
pip install -r requirements.txt

echo ""
echo "----------------------------------------------"
echo "Installing Flash Attention..."
echo "----------------------------------------------"
pip install flash-attn --no-build-isolation

echo ""
echo "----------------------------------------------"
echo "VideoLLaMA3 training environment setup complete!"
echo "----------------------------------------------"
