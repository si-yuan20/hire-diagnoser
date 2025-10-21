##############################
# utils.py
##############################
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import os


from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)

from sklearn.preprocessing import label_binarize


def calculate_auc(targets, probs):
    # 假设是多分类问题，使用roc_auc_score的适当参数
    return roc_auc_score(targets, probs, multi_class='ovr')


def calculate_precision(targets, preds):
    return precision_score(targets, preds, average='macro')


def calculate_recall(targets, preds):
    return recall_score(targets, preds, average='macro')

# def calculate_f1(targets, preds):
#     return f1_score(targets, preds, average='macro')


def calculate_sensitivity(targets, preds):
    """计算多类别敏感度（召回率的类别平均）"""
    cm = confusion_matrix(targets, preds)
    n_classes = cm.shape[0]
    sensitivity = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        sensitivity.append(tp / (tp + fn) if (tp + fn) != 0 else 0.0)
    
    return np.mean(sensitivity)

def calculate_specificity(targets, preds):
    """计算多类别特异性"""
    cm = confusion_matrix(targets, preds)
    n_classes = cm.shape[0]
    specificity = []
    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp)) if (tn + fp) != 0 else 0.0
    return np.mean(specificity)
                           
                           
class MetricTracker:
    def __init__(self):
        self.loss_total = 0.0
        self.total_samples = 0
        self.correct = 0
        self.all_targets = []
        self.all_preds = []
        self.all_probs = []

    def update(self, loss, preds, probs, targets):
        self.loss_total += loss.item() * targets.size(0)
        self.total_samples += targets.size(0)
        self.correct += (preds == targets).sum().item()

        # 收集目标、预测和概率
        self.all_targets.extend(targets.cpu().numpy())
        self.all_preds.extend(preds.cpu().numpy())
        self.all_probs.extend(probs.cpu().detach().numpy())

    def compute(self):
        loss = self.loss_total / self.total_samples
        acc = self.correct / self.total_samples

        # 计算其他指标（如AUC、精确度、召回率）
        auc = calculate_auc(self.all_targets, self.all_probs)
        precision = calculate_precision(self.all_targets, self.all_preds)
        recall = calculate_recall(self.all_targets, self.all_preds)
        f1 = f1_score(self.all_targets, self.all_preds, average='macro')

        sensitivity = calculate_sensitivity(self.all_targets, self.all_preds)
        specificity = calculate_specificity(self.all_targets, self.all_preds)

        return {
            'loss': loss,
            'acc': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'targets': self.all_targets,  # 确保包含这些键
            'preds': self.all_preds,
            'probs': self.all_probs,
            'sensitivity': sensitivity,
            'specificity': specificity,
        }


def plot_learning_curves(train_metrics, val_metrics,dataset_name, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['loss'], label='Train')
    plt.plot(val_metrics['loss'], label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['acc'], label='Train')
    plt.plot(val_metrics['acc'], label='Val')
    plt.title('Accuracy')
    # path = os.path.join(path + dataset_name, model_name)
    plt.savefig(f"result/test_result/{dataset_name}/{model_name}/curves.png")
    plt.close()


def plot_confusion_matrix(targets, preds, class_names, dataset_name,model_name,epoch):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # path = os.path.join('result/test_result/' + dataset_name, model_name)
    plt.savefig(f"result/test_result/{dataset_name}/{model_name}/confusion_matrix_{epoch}.png")
    plt.close()


def plot_roc_curve(targets, probs, class_names,dataset_name, model_name,epoch):
    n_classes = len(class_names)

    # 将标签二值化（例如 [0,1,2] → [[1,0,0], [0,1,0], [0,0,1]]）
    y_true = label_binarize(targets, classes=np.arange(n_classes))

    # 确保概率数组是二维的（样本数 × 类别数）
    y_score = np.array(probs)  # 转换为NumPy数组

    # 计算每个类别的ROC曲线和AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均ROC曲线和AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制所有曲线
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制对角线

    # 绘制每个类别的ROC曲线
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # 绘制宏平均曲线
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro Average (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', lw=2)

    # 设置图表属性
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"result/test_result/{dataset_name}/{model_name}/roc_auc_{epoch}.png")


def grad_cam(model, image, target_class):
    model.eval()
    output = model(image.unsqueeze(0))
    model.zero_grad()
    output[0, target_class].backward()

    gradients = model.gradients['conv']
    activations = model.activations['conv']

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    weighted_activations = torch.zeros_like(activations)
    for i in range(activations.shape[1]):
        weighted_activations[:, i, :, :] = pooled_gradients[i] * activations[:, i, :, :]

    heatmap = torch.mean(weighted_activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap.detach().cpu().numpy()


def plot_feature_heatmaps(images, heatmaps,path='result/test_result/'):
    """
    绘制多尺度特征热力图
    :param images: 原始图像张量 (B, C, H, W)
    :param heatmaps: 各阶段热力图列表 [stage1, stage2, stage3]
    :param path: 保存路径
    """
    plt.figure(figsize=(15, 5))

    # 转换为CPU numpy格式
    img_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    heatmaps = [h.detach().cpu().numpy() for h in heatmaps]

    for i in range(min(3, images.shape[0])):  # 最多显示3个样本
        # 原始图像
        plt.subplot(3, 4, i * 4 + 1)
        plt.imshow(img_np[i])
        plt.title('Original')
        plt.axis('off')

        # 各阶段热力图
        for stage in range(3):
            plt.subplot(3, 4, i * 4 + stage + 2)
            hm = heatmaps[stage][i].mean(0)  # 多通道取均值
            hm = (hm - hm.min()) / (hm.max() - hm.min())  # 归一化
            plt.imshow(hm, cmap='viridis')
            plt.title(f'Stage {stage + 1}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# def plot_learning_rates(history,dataset_name,model_name):
#     plt.figure(figsize=(10, 6))
#     plt.plot(history['lr'], 'o-', label='Learning Rate')
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rate Schedule')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"test_result/{dataset_name}/{model_name}_LR.png")