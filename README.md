# HayStack: When Classical Closed-Form Methods Match Large Pretrained Models for Edge Multimedia Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Paper](https://img.shields.io/badge/paper-IOP%20ERX-green)](https://doi.org/xx.xxxx/erx)

> **HayStack** is a **parameter-free, training-free few-shot classifier** that builds a
> **hybrid subspace** from **K reference images** (K=0--16) using **pyramid pooling** and
> **ResNet18**, **DINO**, **DINOv2**, **CLIP**, and **ViT** backbones.

**Paper**: *HayStack: When Classical Closed-Form Methods Match Large Pretrained Models for Edge Multimedia Recognition*, Engineering Research Express
**Code**: `https://github.com/AnhTus-dey/HayStack.git`  
**Demo**: `scripts/run_benchmark.py`

---

## When Does Scale Help?

This repo accompanies an empirical study identifying **three conditions** under which
web-scale pretraining (CLIP 151M, DINO 22M) provides no measurable advantage over a
lightweight classical baseline:

| Condition | Dataset | Finding |
|-----------|---------|---------|
| **Domain shift** | CheXpert (medical X-ray) | All methods converge to AUROC ~0.5 regardless of model size |
| **Objective mismatch** | CIFAR-100 (fine-grained) | HayStack outperforms TIP-Adapter by >19 pp (69.8% vs 49.6%) |
| **Throughput inversion** | COCO (natural images) | DINO gains 5.6 pp but runs at 3.4x lower throughput (79 vs 267 FPS) |

---

## Key Features

| Feature | Description |
|--------|-------------|
| **Zero-Shot (K=0)** | Uses **learned prior prototype + subspace** |
| **Few-Shot (K=1--16)** | Hybrid PCA + prior (K<3), Pure PCA (K>=3) |
| **14+ Backbones** | ResNet, ViT, DINO, DINOv2, CLIP, MobileNet, EfficientNet |
| **Pyramid Pooling** | Multi-scale spatial aggregation (scales 1, 2, 4) |
| **No Training** | Zero fine-tuning, zero gradients |
| **Robustness** | Noise, blur, brightness, contrast |
| **Reproducible** | 100% deterministic with seed |

---

## Key Results

### Binary Classification (one-vs-rest, CPU inference)

| Method | Params | COCO F1 | CheXpert F1 | CIFAR-100 F1 | FPS (CPU) |
|--------|--------|---------|-------------|--------------|-----------|
| HayStack (RN18) | 11.7M | 65.8% | 27.8% | **69.8%** | **267** |
| RN18 + Cosine (no SVD) | 11.7M | 68.3% | 27.8% | -- | 270 |
| CLIP zero-shot | 151M | 65.5% | 27.7% | 24.2% | 145 |
| CLIP + Cosine (no SVD) | 151M | 55.0% | 27.8% | -- | 12 |
| HayStack-on-CLIP | 151M | 57.2% | -- | -- | 12 |
| TIP-Adapter | 151M | 71.5% | 27.8% | 49.6% | 195 |
| DINO (ViT-S/16) | 22M | **71.4%** | 26.0% | 79.6% | 79 |

> Best K per method shown. HayStack-on-CLIP applies the SVD subspace on CLIP features (diagnostic ablation only).

### CheXpert: All Methods Converge to Random (AUROC ~0.5)

| Method | K | AUROC |
|--------|---|-------|
| HayStack (RN18, 11.7M) | 8 | 0.488 +- 0.012 |
| CLIP zero-shot (151M) | 0 | 0.489 +- 0.011 |
| TIP-Adapter (151M) | 8 | 0.480 +- 0.011 |
| DINO (22M) | 8 | 0.467 +- 0.010 |
| Random baseline | -- | 0.500 |

> AUROC is threshold-independent. Convergence to 0.5 confirms domain shift as the operative
> failure mode, not method quality.

### CIFAR-100: Objective Mismatch vs Data Coverage

| Method | K=3 | K=5 | K=8 | K=16 |
|--------|-----|-----|-----|------|
| HayStack (RN18) | 82.0% | 83.8% | 84.7% | 86.8% |
| RN18 + Cosine (no SVD) | 83.7% | 85.5% | 86.6% | 87.4% |
| CLIP zero-shot | 51.5% | 53.2% | 51.4% | 53.7% |
| CLIP linear probe | 80.2% | 85.0% | 90.0% | 90.8% |

> At K=3, HayStack (82.0%) outperforms CLIP linear probe (80.2%), confirming the
> low-shot edge advantage. At K>=5, CLIP linear probe surpasses HayStack.

### Multi-Class Stability (10 independent runs)

| Dataset | K=3 | K=5 | K=8 | Random |
|---------|-----|-----|-----|--------|
| COCO 5-way | 49.3% +- 8.8% | 56.0% +- 5.7% | 61.8% +- 4.6% | 20% |
| CheXpert 3-way | 38.9% +- 6.1% | 40.0% +- 6.3% | 44.8% +- 8.7% | 33% |
| CIFAR-10 5-way | 67.6% +- 5.0% | 70.7% +- 6.1% | 74.9% +- 5.1% | 20% |

> High variance at K=3 COCO (std=8.8%) reflects sensitivity to reference selection
> in underdetermined subspaces. Use K>=5 for multi-class tasks.

---

## Installation

```bash
git clone https://github.com/AnhTus-dey/HayStack.git
cd HayStack
pip install -e ".[clip]"   # Includes CLIP (optional)
```

> **Note**: Use `[clip]` only if you want to run CLIP baselines.

---

## Directory Structure

```
HayStack/
├── haystack/              # Core model + utils
├── datasets/              # COCO, CheXpert, CIFAR-10 loaders
├── baselines/             # CLIP, DINO, SimCLR, ProtoNet
├── robustness/            # Corruption transforms
├── scripts/               # run_benchmark.py
├── demo/                  # zeroshot_demo.ipynb
├── results/               # Auto-generated JSON outputs
├── requirements.txt
└── README.md
```

---

## Quick Start (5 Minutes)

### 1. Download COCO 2017 (val only, ~1.2 GB)

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -O data/coco/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations.zip

unzip data/coco/val2017.zip -d data/coco/
unzip data/coco/annotations.zip -d data/coco/
rm data/coco/*.zip
```

### 2. Run HayStack on COCO (Person Detection)

```bash
python scripts/run_benchmark.py \
  --dataset coco \
  --data_dir data/coco \
  --K 5 \
  --n_test 1000 \
  --device cuda
```

**Expected Output**:
```
HayStack | K=5 | F1: 65.8% | FPS: 267 | Dim: 80
```

---

## Full Benchmark Suite

Run **all experiments** (zero-shot, few-shot, multi-class, robustness, 14 backbones):

```bash
python scripts/run_benchmark.py \
  --run_all \
  --data_dir data \
  --device cuda \
  --seed 42
```

> Outputs saved to `results/comprehensive_results.json`

---

## Supported Backbones

```python
[
    'resnet18', 'resnet50',
    'mobilenet_v2', 'efficientnet_b0',
    'vit_b_16', 'swin_t',
    'dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16',
    'dinov2_vits14', 'dinov2_vitb14',
    'clip_ViT-B/16', 'clip_ResNet50'
]
```

| Backbone | Params | COCO F1 | FPS (CPU) | Use case |
|----------|--------|---------|-----------|----------|
| ResNet18 | 11.7M | 65.5% | 267 | **Edge deployment (recommended)** |
| MobileNetV2 | 3.4M | 68.0% | 138 | Ultra-low memory |
| ResNet50 | 25.6M | 69.1% | 66 | Accuracy-speed balance |
| Swin-T | 28M | 69.6% | 48 | High accuracy, server |
| DINO ViT-S/16 | 22M | 71.3% | 72 | Best accuracy/throughput |
| DINO ViT-S/8 | 22M | 73.4% | 13 | Highest accuracy, offline only |

---

## Zero-Shot Demo (K=0)

```python
from haystack.model import HayStackMultiBackbone
import torch

model = HayStackMultiBackbone(backbone_name='resnet18', device='cuda').eval()

# Zero-shot inference (no reference images!)
with torch.no_grad():
    Psi, dim, U = model.build_psi(None, K=0)
    R = model.extract_R(image_tensor)
    score = (Psi * (R @ U)).sum()
```

> **Note**: Replace random prior with trained prior for real zero-shot performance.

---

## Command Line Options

```bash
python scripts/run_benchmark.py --help
```

```
Options:
  --dataset       [coco|chexpert|cifar10]     Dataset to evaluate
  --data_dir      Path to dataset root
  --K             Number of reference shots (0-16)
  --n_test        Number of test samples
  --device        [cuda|cpu]
  --run_all       Run full benchmark suite
  --seed          Random seed (default: 42)
```

---

## Configuration Guide

| Deployment scenario | Recommended config |
|--------------------|--------------------|
| Unknown / multi-domain | HayStack-General (all components) |
| Visual surveillance | HayStack-General (handles diversity) |
| Medical / X-ray | HayStack-Fast (no jitter) |
| Fine-grained classification | HayStack-Fast (no jitter) |
| Industrial inspection | HayStack-Fast (controlled environment) |

**HayStack-General**: pyramid pooling + adaptive dim + hybrid prior + jitter  
**HayStack-Fast**: pyramid pooling + adaptive dim + hybrid prior (no jitter)

---

## Hybrid YOLO + HayStack Pipeline

For surveillance applications, HayStack filters false positives from a YOLO detector:

| System | Precision | Recall | F1 | False Positives | FPS |
|--------|-----------|--------|----|-----------------|-----|
| YOLOv8n alone | 84.6% | 75.9% | 80.1% | 2 | 185 |
| YOLO + HayStack | **100.0%** | 91.7% | **95.7%** | **0** | 142 |

Zero false positives from 2 reference images, operational immediately without labelled training data.

---

## Reproducibility

```bash
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

python scripts/run_benchmark.py --run_all --seed 42 --device cuda
```

95% bootstrap confidence intervals computed with 100 resamples.
Threshold selected via Youden's J-statistic on a held-out validation split
(60/20/20 train-val-test, stratified per class, disjoint from reference and test sets).

---

## Limitations

- **Diffuse patterns** (e.g., smoke): 25.0% F1 due to high intra-class variability; temporal context needed.
- **Low-shot multi-class** (K=3, 5-way): high variance (std=8.8%); use K>=5 for multi-class tasks.
- **Threshold calibration**: optimal threshold requires 10--20 validation images per class.
- **Domain shift**: on out-of-distribution medical data, all methods converge to AUROC ~0.5 -- a fundamental limitation of appearance-based methods, not specific to HayStack.

---

## Citation

```bibtex
@article{nguyen2025haystack,
  title   = {HayStack: When Classical Closed-Form Methods Match Large Pretrained Models
             for Edge Multimedia Recognition},
  author  = {Nguyen, Tu Anh and Nguyen, Huy Hoang and Nguyen, Vinh Dinh},
  journal = {Engineering Research Express},
  year    = {2026},
  note    = {Accepted}
}
```

---

## License

[MIT License](LICENSE) -- Free for research and commercial use.

---

## Star this repo if you find it useful!

---

## Acknowledgments

- COCO dataset: https://cocodataset.org
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert
- DINO / DINOv2: https://github.com/facebookresearch/dino
- CLIP: https://github.com/openai/CLIP
- PyTorch: https://pytorch.org
