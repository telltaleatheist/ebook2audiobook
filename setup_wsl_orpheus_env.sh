#!/bin/bash
# =============================================================================
# Minimal Orpheus TTS Environment Setup for WSL
# =============================================================================
# Creates a clean environment optimized for Orpheus TTS with CUDA graphs
# (NO enforce_eager needed - full CUDA graph performance)
#
# Based on research:
# - vLLM 0.7.3 (recommended by Orpheus authors)
# - PyTorch 2.5.1 + CUDA 12.1
# - Minimal dependencies to avoid conflicts
#
# Key fix: orpheus.py now sets these env vars to enable CUDA graphs:
#   TORCH_CUDA_ENABLE_CUDA_GRAPH=1
#   CUDA_LAUNCH_BLOCKING=0
#
# Usage:
#   chmod +x setup_wsl_orpheus_env.sh
#   ./setup_wsl_orpheus_env.sh
# =============================================================================

set -e

ENV_NAME="orpheus_tts"
PYTHON_VERSION="3.11"

echo "=============================================="
echo "  Minimal Orpheus TTS Environment for WSL"
echo "=============================================="
echo ""
echo "This will create a clean environment with:"
echo "  - Python ${PYTHON_VERSION}"
echo "  - PyTorch 2.5.1 + CUDA 12.1"
echo "  - vLLM 0.7.3 (with CUDA graph support)"
echo "  - orpheus-speech + snac"
echo ""

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA may not be available in WSL."
    echo "Make sure you have:"
    echo "  1. NVIDIA GPU drivers installed on Windows"
    echo "  2. WSL2 with CUDA support enabled"
    exit 1
fi

echo "CUDA detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Remove old environment if exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y
fi

# Create fresh environment
echo ""
echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "---"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate the environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch 2.5.1 with CUDA 12.1
echo ""
echo "Installing PyTorch 2.5.1 with CUDA 12.1..."
echo "---"
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: CUDA not available!')
    exit(1)
"

# Install vLLM 0.7.3 (recommended by Orpheus authors)
echo ""
echo "Installing vLLM 0.7.3..."
echo "---"
pip install vllm==0.7.3

# Install transformers (needed for tokenizer)
echo ""
echo "Installing transformers..."
pip install transformers

# Install orpheus-speech
echo ""
echo "Installing orpheus-speech..."
echo "---"
pip install orpheus-speech

# Install snac explicitly (audio decoder)
echo ""
echo "Installing snac (audio codec)..."
pip install snac

# Install additional dependencies for ebook2audiobook worker
echo ""
echo "Installing additional ebook2audiobook dependencies..."
pip install soundfile scipy numpy

# Verify all installations
echo ""
echo "=============================================="
echo "  Verifying Installation"
echo "=============================================="
python -c "
import torch
import vllm
import snac
from orpheus_tts import OrpheusModel
from transformers import AutoTokenizer

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'vLLM: {vllm.__version__}')
print(f'SNAC: OK')
print(f'Orpheus TTS: OK')
print(f'Transformers: OK')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Environment: ${ENV_NAME}"
echo ""
echo "To activate: conda activate ${ENV_NAME}"
echo ""
echo "Quick test (will download model on first run ~6GB):"
echo "  python -c \""
echo "from orpheus_tts import OrpheusModel"
echo "m = OrpheusModel(model_name='canopylabs/orpheus-3b-0.1-ft')"
echo "print('Model loaded with CUDA graphs!')"
echo "\""
echo ""
echo "NOTE: This environment is optimized for Orpheus TTS only."
echo "      CUDA graphs should work without enforce_eager."
