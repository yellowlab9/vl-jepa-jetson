"""
Setup script for Jetson Orin Nano
Configures power mode, installs dependencies, and sets up environment
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """Run shell command"""
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def check_jetson():
    """Check if running on Jetson"""
    jetson_model_file = Path("/etc/nv_tegra_release")
    if jetson_model_file.exists():
        print("Detected NVIDIA Jetson platform")
        with open(jetson_model_file) as f:
            print(f.read())
        return True
    else:
        print("Warning: Not running on Jetson platform")
        return False


def set_power_mode(mode="15W"):
    """Set Jetson power mode"""
    print(f"\nSetting power mode to {mode}...")
    
    # Map mode to nvpmodel value
    mode_map = {
        "7W": "1",
        "15W": "0",
        "MAXN": "0",
    }
    
    mode_value = mode_map.get(mode, "0")
    
    # Set power mode
    run_command(["sudo", "nvpmodel", "-m", mode_value], check=False)
    
    # Max clocks
    run_command(["sudo", "jetson_clocks"], check=False)
    
    # Show current mode
    run_command(["sudo", "nvpmodel", "-q"], check=False)


def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Upgrade pip
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("Warning: requirements.txt not found")
        
        # Install core packages
        packages = [
            "transformers",
            "timm",
            "pillow",
            "opencv-python",
            "einops",
            "wandb",
            "tensorboard",
            "tqdm",
            "pyyaml",
            "omegaconf",
            "pycocotools",
        ]
        
        for pkg in packages:
            run_command([sys.executable, "-m", "pip", "install", pkg])
        
        # Try to install bitsandbytes (might need special build for Jetson)
        print("\nTrying to install bitsandbytes...")
        run_command([sys.executable, "-m", "pip", "install", "bitsandbytes"], check=False)


def check_pytorch():
    """Check PyTorch installation"""
    print("\nChecking PyTorch installation...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Warning: CUDA not available!")
        
        return True
    except ImportError:
        print("Error: PyTorch not installed!")
        print("Please install PyTorch for Jetson from:")
        print("https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
        return False


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")
    
    dirs = [
        "data",
        "checkpoints",
        "logs",
        "runs",
    ]
    
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"Created: {d}/")


def download_data_info():
    """Print data download instructions"""
    print("\n" + "="*80)
    print("DATA DOWNLOAD INSTRUCTIONS")
    print("="*80)
    
    print("\nCOCO Captions:")
    print("1. Download from: https://cocodataset.org/#download")
    print("2. Required files:")
    print("   - train2017.zip (images)")
    print("   - val2017.zip (images)")
    print("   - annotations_trainval2017.zip (annotations)")
    print("3. Extract to: ./data/")
    print("   Expected structure:")
    print("   data/")
    print("   ├── images/")
    print("   │   ├── train2017/")
    print("   │   └── val2017/")
    print("   └── annotations/")
    print("       ├── captions_train2017.json")
    print("       └── captions_val2017.json")
    
    print("\nFlickr30k (Alternative):")
    print("1. Request access: https://shannon.cs.illinois.edu/DenotationGraph/")
    print("2. Download images and captions")
    print("3. Extract to: ./data/flickr30k/")


def main():
    print("="*80)
    print("VL-JEPA Setup for Jetson Orin Nano")
    print("="*80)
    
    # Check if on Jetson
    is_jetson = check_jetson()
    
    # Check PyTorch
    if not check_pytorch():
        print("\nPlease install PyTorch first, then run this script again.")
        return
    
    # Set power mode (only on Jetson)
    if is_jetson:
        set_power_mode("15W")
    
    # Install dependencies
    install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Print data download info
    download_data_info()
    
    print("\n" + "="*80)
    print("Setup completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Download COCO Captions dataset (see instructions above)")
    print("2. Update config.yaml if needed")
    print("3. Run training: python train.py --config config.yaml")
    print("\nFor monitoring:")
    print("- Memory usage: watch -n 1 tegrastats")
    print("- GPU usage: watch -n 1 nvidia-smi")
    print("="*80)


if __name__ == "__main__":
    main()
