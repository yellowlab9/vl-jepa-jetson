#!/bin/bash
# Install correct PyTorch for Jetson Orin Nano
# This script installs NVIDIA's official PyTorch wheels for JetPack

set -e

echo "=============================================="
echo "Installing PyTorch with CUDA for Jetson"
echo "=============================================="

# Check JetPack version
echo "Checking JetPack version..."
cat /etc/nv_tegra_release 2>/dev/null || echo "Could not read tegra release"

# Set LD_LIBRARY_PATH for CUDA
echo ""
echo "Setting up CUDA environment..."
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Add to bashrc for persistence
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA environment for Jetson" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
    echo "Added CUDA paths to ~/.bashrc"
fi

# Uninstall CPU-only PyTorch
echo ""
echo "Removing CPU-only PyTorch..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Detect JetPack version and install correct PyTorch
# JetPack 5.x (L4T R35.x) uses Python 3.8 or 3.10
# JetPack 6.x (L4T R36.x) uses Python 3.10 or 3.11

echo ""
echo "Installing PyTorch from NVIDIA's Jetson repository..."

# Method 1: Use pip with NVIDIA's index (preferred for JetPack 5.1+)
pip3 install --upgrade pip

# For JetPack 5.1.2+ / L4T R35.4.1+
# PyTorch 2.1 with CUDA support
pip3 install numpy==1.26.4

# Install from NVIDIA's wheel server
pip3 install --no-cache-dir torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60

# If above fails, try JetPack 5.x wheels
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "JetPack 6.x wheels failed, trying JetPack 5.x..."
    pip3 uninstall -y torch torchvision 2>/dev/null || true
    pip3 install --no-cache-dir torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v512
fi

# If pip install still fails, download manually
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Pip install failed, downloading wheel manually..."
    
    # Get Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python version: $PYTHON_VERSION"
    
    # For JetPack 5.1.2 with Python 3.10
    if [ "$PYTHON_VERSION" == "3.10" ]; then
        # PyTorch 2.1.0 for JetPack 5.1.2
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"
        TORCHVISION_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torchvision-0.16.0a0+7a3b7ba-cp310-cp310-linux_aarch64.whl"
        
        wget -q --show-progress -O torch.whl "$TORCH_URL" || true
        wget -q --show-progress -O torchvision.whl "$TORCHVISION_URL" || true
        
        if [ -f torch.whl ]; then
            pip3 install torch.whl torchvision.whl
            rm -f torch.whl torchvision.whl
        fi
    fi
fi

# Final verification
echo ""
echo "=============================================="
echo "Verifying PyTorch CUDA installation..."
echo "=============================================="

python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Quick test
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print("CUDA computation test: PASSED ✓")
else:
    print("ERROR: CUDA still not available!")
    print("Please install PyTorch manually from:")
    print("https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
    exit(1)
EOF

echo ""
echo "PyTorch CUDA installation complete!"
