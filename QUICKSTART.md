# Quick Start Guide - VL-JEPA on Jetson Orin Nano

## 🚀 Fast Setup (10 minutes)

### 1. Install & Setup
```bash
# Setup environment
python scripts/setup_jetson.py

# Test installation
python scripts/test_implementation.py
```

### 2. Download Sample Data (COCO - subset)
```bash
# Create data directory
mkdir -p data

# Download COCO 2017 (adjust paths as needed)
# Option A: Full dataset (~20GB)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip -d data/images/
unzip val2017.zip -d data/images/
unzip annotations_trainval2017.zip -d data/
```

### 3. Quick Training Test (10K samples)
```bash
# Edit config.yaml to limit dataset
# Set max_samples: 10000

# Start training
python train.py --config config.yaml
```

## ⚡ Quick Commands

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Monitor Jetson stats
tegrastats

# Check power mode
sudo nvpmodel -q

# Set to 15W mode
sudo nvpmodel -m 0
sudo jetson_clocks

# View logs
tail -f logs/train_*.log

# Resume training
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

## 📊 Expected Results

### After 10 epochs (~2-3 hours on 10K samples):
- Training loss: ~0.5-1.0
- Memory usage: 2-3GB
- Throughput: 2-5 samples/sec

### After 50 epochs (~12-18 hours on 10K samples):
- Training loss: ~0.2-0.5
- Retrieval metrics: Starting to show learning
- Model saved in checkpoints/

## 🎯 Configuration Presets

### Minimal (fastest, for testing)
```yaml
data:
  max_samples: 1000
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  num_epochs: 10
```

### Balanced (recommended)
```yaml
data:
  max_samples: 10000
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  num_epochs: 50
```

### Full (best results, slow)
```yaml
data:
  max_samples: null  # Use all data
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  num_epochs: 100
```

## 🔧 Common Issues

### OOM Error
```bash
# Reduce batch size
training:
  batch_size: 1
```

### Too Slow
```bash
# Reduce dataset
data:
  max_samples: 5000
```

### Can't Find CUDA
```bash
# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch for Jetson
# See: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

## 📈 Next Steps

1. **Train on full dataset** (2-4 days)
2. **Evaluate on downstream tasks**
3. **Export for deployment**
4. **Optimize with TensorRT**

## 🆘 Need Help?

1. Check README.md for full documentation
2. Run test suite: `python scripts/test_implementation.py`
3. View logs: `tail -f logs/train_*.log`
4. Open GitHub issue

## ✅ Checklist

- [ ] Setup completed
- [ ] Tests passed
- [ ] Data downloaded
- [ ] Training started
- [ ] Model checkpointed
- [ ] Evaluation run

Happy training! 🎉
