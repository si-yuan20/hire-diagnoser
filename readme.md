```markdown
# Hires-Diagnoser: Dual-Stream Medical Image Diagnosis Framework Based on Multi-Level Resolution Adaptive Sensing

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch Version](https://img.shields.io/badge/pytorch-1.12%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This project implements the dual-stream medical image diagnosis framework proposed in the paper "Hires-Diagnoser: A dual stream medical image diagnosis framework based on multi-level resolution adaptive sensing", combining the advantages of ConvNeXt and Swin-Transformer to achieve multi-scale feature fusion and adaptive perception.

## 📋 Project Overview

Hires-Diagnoser is an advanced medical image classification framework with the following core features:

- **Dual-stream parallel architecture**: Simultaneously utilizes ConvNeXt for local texture feature extraction and Swin-Transformer for global context dependency capture
- **Multi-level resolution perception**: Feature interaction at three resolution levels: 56×56, 28×28, and 14×14
- **Adaptive feature fusion**: Dynamic cross-modal feature fusion through LCA (Light Cross-Attention) module
- **Efficient diagnosis solution**: Achieves SOTA performance on multiple medical image datasets

## 📁 Data & Weights Download

### Dataset Download
All experimental datasets can be obtained through the following link:
```bash
Link: https://pan.baidu.com/s/1RkIz2Utt0vTkpkPyjTN27A?pwd=9563 
Extraction code: 9563
```

After downloading, please organize the data according to the following directory structure:
```bash
datasets/
├── Raabin-WBC/
│   ├── Basophil/
│   ├── Eosinophil/
│   ├── Lymphocyte/
│   ├── Monocyte/
│   └── Neutrophil/
├── Brain_Tumor_MRI/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Notumor/
│   └── Pituitary/
├── LC25000/
│   ├── Colon-A/
│   ├── Benign-C/
│   ├── Lung-A/
│   ├── Lung-S/
│   └── Benign-L/
└── OCT-C8/
    ├── AMD/
    ├── CNV/
    ├── CSR/
    ├── DME/
    ├── DR/
    ├── MH/
    ├── DRUSEN/
    └── NORMAL/
```

### Pre-trained Weights
Pre-trained ConvNeXt and Swin-Transformer weights will be automatically downloaded to the `~/.cache/torch/hub/checkpoints/` directory.

## 🚀 Quick Start

### Environment Setup
```bash
# Clone the project
git clone https://github.com/your-username/Hires-Diagnoser.git
cd Hires-Diagnoser

# Install dependencies
pip install -r requirements.txt
```

### Training Commands
```bash
# Basic training
python train.py \
  --data_dir /path/to/dataset \
  --dataset Raabin-WBC \
  --batch_size 64 \
  --epochs 50 \
  --lr_convnext 1e-4 \
  --lr_swin 1e-5 \
  --lr_fusion 1e-3

# Using attention mechanism
python train.py \
  --data_dir /path/to/dataset \
  --dataset Brain_Tumor_MRI \
  --attention cbam \
  --batch_size 32
```

### Testing Commands
```bash
python test.py \
  --data_dir /path/to/dataset \
  --dataset LC25000 \
  --checkpoint /path/to/checkpoint.pth
```

## 🧠 Core Features

### Network Architecture Characteristics

- **Dual-stream feature extraction**
  ```python
  # ConvNeXt branch - local feature extraction
  convnext_features = convnext_backbone(images)
  
  # Swin-Transformer branch - global context
  swin_features = swin_backbone(images)
  ```

- **Multi-level resolution LCA fusion**
  ```python
  # Feature fusion at three resolution levels
  fused_56 = LCA_module(convnext_56, swin_56)
  fused_28 = LCA_module(convnext_28, swin_28)
  fused_14 = LCA_module(convnext_14, swin_14)
  ```

### Optimization Strategy

| Component | Learning Rate | Optimizer | Weight Decay |
|-----------|---------------|-----------|--------------|
| ConvNeXt Branch | 1e-4 | AdamW | 1e-4 |
| Swin Branch | 1e-5 | AdamW | 1e-4 |
| Fusion Module | 1e-3 | SGD | 1e-4 |

### Supported Datasets

```python
DATASETS = {
    'Raabin-WBC': ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil'],
    'Brain_Tumor_MRI': ['Glioma', 'Meningioma', 'Notumor', 'Pituitary'],
    'LC25000': ['Colon-A', 'Benign-C', 'Lung-A', 'Lung-S', 'Benign-L'],
    'OCT-C8': ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'MH', 'DRUSEN', 'NORMAL']
}
```

## 📊 Performance Metrics

Test results on four benchmark datasets:

| Dataset | Accuracy | Precision | F1-Score | Recall |
|---------|----------|-----------|----------|--------|
| Raabin-WBC | 98.59% | 96.18% | 96.36% | 96.27% |
| Brain Tumor MRI | 95.45% | 95.57% | 95.26% | 95.19% |
| LC25000 | 99.43% | 99.44% | 99.43% | 99.43% |
| OCT-C8 | 95.23% | 95.55% | 95.35% | 95.23% |

## 📝 Citation

If this project is helpful for your research, please cite our paper:

```bibtex
@article{zhao2025hires,
  title={Hires-Diagnoser: A dual stream medical image diagnosis framework based on multi-level resolution adaptive sensing},
  author={Zhao, Si-chao and Chen, Jun-jun and Shi, Shi-long and Deng, Ge and Qiu, Xue-jun},
  journal={Biomedical Physics & Engineering Express},
  year={2025}
}
```