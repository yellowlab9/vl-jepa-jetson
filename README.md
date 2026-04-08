# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

[![GitHub](https://img.shields.io/badge/GitHub-vl--jepa--jetson-blue?logo=github)](https://github.com/mandarwagh9/vl-jepa-jetson)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin-nano)

This is a fork of the **Implementation of VL-JEPA optimized for **NVIDIA Jetson Orin Nano (8GB)** with memory-efficient training using FP16, gradient checkpointing, and 8-bit optimizers. In this fork, missing files from the data module are added. The missing data module is generated
by Gemini 3.1 Pro.


**🔗 Original repository**: https://github.com/mandarwagh9/vl-jepa-jetson   
**🔗 Repository with missing data module added**: https://github.com/yellowlab9/vl-jepa-jetson

## 🚀 Features

- **Lightweight Architecture**: ViT-Tiny (192-dim) + DistilBERT (768-dim) + MLP Predictor
- **Memory Optimized**: FP16 mixed precision, gradient checkpointing, 8-bit AdamW
- **JEPA Training**: Self-supervised learning with masked patch prediction
- **Jetson Ready**: Optimized for 8GB unified memory, ~2-3GB training footprint
- **Multi-Modal**: Vision-language pretraining with COCO Captions

## 📋 Requirements

### Hardware
- NVIDIA Jetson Orin Nano (8GB)
- MicroSD card (64GB+ recommended)
- Power supply (15W mode recommended)

### Software
- JetPack 5.x or 6.x (includes PyTorch)
- Python 3.8+
- CUDA 11.4+

## 🛠️ Installation

### 1. Setup Jetson Environment

```bash
# Clone repository
git clone https://github.com/mandarwagh9/vl-jepa-jetson.git
cd vl-jepa-jetson

# Run setup script
python scripts/setup_jetson.py
```

This will:
- Set power mode to 15W
- Install Python dependencies
- Create necessary directories
- Check PyTorch installation

### 2. Manual Installation (if needed)

```bash
# Install dependencies
pip3 install -r requirements.txt

# Install 8-bit optimizers (optional, for memory savings)
pip3 install bitsandbytes

# Create directories
mkdir -p data checkpoints logs runs
```

## 📊 Dataset Preparation

### COCO Captions (Recommended)

1. Download from [COCO Dataset](https://cocodataset.org/#download):
   - `train2017.zip` (~18GB)
   - `val2017.zip` (~1GB)
   - `annotations_trainval2017.zip` (~241MB)

2. Extract to `./data/`:
```bash
cd data
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ..
```

3. Expected structure:
```
data/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

### Dataset Subset (For Testing)

To limit dataset size (recommended for initial testing):

```yaml
# In config.yaml
data:
  max_samples: 10000  # Use only 10K samples
```

## 🎯 Training

### Basic Training

```bash
# Start training
python train.py --config config.yaml

# With Weights & Biases logging
python train.py --config config.yaml --wandb

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

### Configuration

Edit `config.yaml` to adjust:
- Model size (ViT-Tiny by default)
- Batch size (2 by default, with grad accumulation=8)
- Learning rate (1e-4 by default)
- Number of epochs (100 by default)
- Dataset size limit

### Expected Training Time

| Dataset Size | Epochs | Time (15W mode) |
|-------------|--------|-----------------|
| 10K samples | 50     | ~12-18 hours    |
| 50K samples | 100    | ~2-4 days       |
| 118K (full) | 100    | ~4-7 days       |

### Memory Usage

- **Training**: ~2-3GB GPU memory
- **Inference**: ~1-2GB GPU memory
- **Dataset**: ~20-25GB disk space (COCO)

## 📈 Monitoring

### Real-time Monitoring

```bash
# GPU memory and utilization
watch -n 1 nvidia-smi

# Jetson-specific stats (temperature, power, memory)
tegrastats

# Check power mode
sudo nvpmodel -q

# TensorBoard
tensorboard --logdir runs/
```

### Weights & Biases

Configure in `config.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "vl-jepa-jetson"
  wandb_entity: "your-username"
```

## 🧪 Model Architecture

```
VL-JEPA Model (~80M parameters)
├── Vision Encoder (ViT-Tiny)
│   ├── Patch Embedding: 16x16 patches
│   ├── Hidden Dim: 192
│   ├── Layers: 12
│   ├── Attention Heads: 3
│   └── Parameters: ~5.7M
├── Text Encoder (DistilBERT)
│   ├── Hidden Dim: 768
│   ├── Layers: 6
│   ├── Max Length: 128 tokens
│   └── Parameters: ~66M
├── Predictor (MLP)
│   ├── Input: 192
│   ├── Hidden: 256
│   ├── Output: 192
│   ├── Layers: 3
│   └── Parameters: ~5M
└── Target Encoders (EMA)
    └── Momentum: 0.996 → 1.0
```

## 🎨 Masking Strategy

JEPA-style multi-block masking:
- **Context blocks**: 1 block (85-100% scale) - what model sees
- **Target blocks**: 4 blocks (15-20% scale) - what model predicts
- **Patch grid**: 14×14 = 196 patches
- **No overlap**: Targets don't overlap with context

## 📊 Evaluation

```bash
# Run evaluation only
python train.py --config config.yaml --resume checkpoints/best_model.pth --eval_only
```

Metrics:
- Image-to-text retrieval (R@1, R@5, R@10)
- Text-to-image retrieval (R@1, R@5, R@10)
- Mean recall

## 🔧 Troubleshooting

### Out of Memory

1. Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 1  # Reduce from 2 to 1
```

2. Increase gradient accumulation:
```yaml
training:
  gradient_accumulation_steps: 16  # Increase from 8
```

3. Clear cache more frequently:
```yaml
training:
  empty_cache_every: 50  # Reduce from 100
```

### Slow Training

1. Check power mode:
```bash
sudo nvpmodel -m 0  # 15W mode
sudo jetson_clocks   # Max clocks
```

2. Reduce dataset size:
```yaml
data:
  max_samples: 10000
```

3. Reduce data workers:
```yaml
data:
  num_workers: 2  # Reduce if bottlenecked
```

### Model Not Learning

1. Check learning rate:
```yaml
training:
  learning_rate: 1.0e-4  # Try 1e-3 or 5e-5
```

2. Verify dataset loading:
```python
# Test dataset
python -c "from vl_jepa.data.dataset import create_dataset; print('Dataset OK')"
```

3. Check masking:
```python
# Visualize masks
from vl_jepa.masks.multiblock import MultiBlockMaskGenerator
mask_gen = MultiBlockMaskGenerator()
ctx, tgt = mask_gen()
print(f"Context: {ctx.sum()}, Target: {tgt.sum()}")
```

## 🚀 Deployment

### Export for Inference

```python
import torch
from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.utils.config import load_config
from vl_jepa.utils.checkpoint import load_model_only

# Load model
config = load_config('config.yaml')
model = create_vl_jepa_model(config)
load_model_only(model, 'checkpoints/best_model.pth')
model.eval()

# Save for deployment
torch.save(model.state_dict(), 'vl_jepa_deployment.pth')
```

### TensorRT Optimization (Future)

For faster inference:
```yaml
jetson:
  tensorrt:
    enabled: true
    precision: "fp16"
    workspace_size: "2GB"
```

## 📚 Project Structure

```
VL-JEPA-T1/
├── vl_jepa/                  # Main package
│   ├── models/              # Model implementations
│   │   ├── vision_encoder.py   # ViT-Tiny
│   │   ├── text_encoder.py     # DistilBERT
│   │   ├── predictor.py        # MLP/Transformer
│   │   └── vl_jepa.py          # Main model
│   ├── data/                # Data loading
│   │   ├── dataset.py          # COCO/Flickr datasets
│   │   ├── transforms.py       # Augmentations
│   │   └── collate.py          # Batch collation
│   ├── masks/               # Masking strategy
│   │   └── multiblock.py       # Multi-block masks
│   └── utils/               # Utilities
│       ├── config.py           # Config management
│       ├── logger.py           # Logging
│       ├── checkpoint.py       # Checkpointing
│       └── metrics.py          # Evaluation metrics
├── scripts/                 # Utility scripts
│   └── setup_jetson.py         # Jetson setup
├── train.py                 # Training script
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Based on [I-JEPA](https://arxiv.org/abs/2301.08243) by Meta AI
- Inspired by [V-JEPA](https://arxiv.org/abs/2401.08377)
- Built for [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-orin-nano)

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🔗 References

- [VL-JEPA Paper](https://arxiv.org/abs/2512.10942v1)
- [I-JEPA: A Path Towards Autonomous Machine Intelligence](https://arxiv.org/abs/2301.08243)
- [COCO Dataset](https://cocodataset.org/)
- [Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)

---

**Note**: This is an educational implementation optimized for Jetson Orin Nano. For production use or larger-scale training, consider using multi-GPU systems with more memory.
