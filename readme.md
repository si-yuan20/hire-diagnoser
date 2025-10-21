```markdown
# 多模态医学影像分类框架

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch Version](https://img.shields.io/badge/pytorch-1.12%2B-orange)

本项目为医学影像分类提供先进的深度学习解决方案，支持多模态特征融合和混合优化策略。

## 📂 数据集结构
```bash
dataset_root/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── class2/
    ├── img1.jpg
    └── ...
```

## 🚀 快速开始
### 训练命令
```bash
python main.py \
  --data_dir /path/to/dataset \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --model convnext_swin
```

### 核心参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | required | 数据集根目录 |
| `--batch_size` | 64 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 基础学习率 |
| `--model` | convnext_swin | 模型架构选择 |

## 🧠 核心功能

### 网络架构特性
- **多尺度特征融合**  
  通过`LAF`（层级注意力融合）和`LCA`（跨模态关联）模块实现动态特征交互
  ```python
  # 特征融合示例
  fused_feature = LAF(layer1_feat, layer2_feat)
  cross_modal_feat = LCA(image_feat, text_feat)
  ```

### 优化策略
| 组件 | 学习率 | 优化器 | 说明 |
|------|--------|--------|------|
| ConvNeXt | 1e-4 | AdamW | 分层学习率衰减 |
| Swin | 1e-5 | Lion | 混合精度训练 |

### 评估指标
```python
metrics = {
    'Accuracy': 0.94,
    'AUC': 0.96,
    'F1': 0.92,
    'Sensitivity': 0.89,
    'Specificity': 0.97
}
```

## 📊 可视化功能
1. **训练监控**
   - 实时Loss/Accuracy曲线
   - 学习率变化趋势

2. **结果分析**
   ```python
   # 生成混淆矩阵
   plot_confusion_matrix(y_true, y_pred)
   
   # 绘制ROC曲线
   plot_roc_curve(y_true, probas)
   ```

3. **可解释性分析**
   ```bash
   python grad_cam.py --img_path sample.jpg --layer_name layer4
   ```
   ![Grad-CAM示例](images/cam_demo.png)
```
---

**提示**：使用前请确保满足以下依赖：
- CUDA 11.7+
- PyTorch 1.12+
- OpenCV 4.6+
```