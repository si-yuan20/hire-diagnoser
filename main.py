##############################
# main.py
##############################
import torch
import argparse
from model import MedicalNet
from data_prepare import create_loaders
from utils import *
from torch.optim import AdamW, SGD
import torch.nn.functional as F
import os
from torch import nn
from tqdm import tqdm  # 添加进度条库
import time
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR  # 新增余弦衰减策略

# 定义颜色代码
COLOR = {
    'HEADER': '\033[95m',
    'TRAIN': '\033[94m',
    'VAL': '\033[93m',
    'TEST': '\033[92m',
    'ENDC': '\033[0m'
}

# Suppress all warnings
warnings.filterwarnings("ignore")


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,dataset_name,model_name):
    model.train()
    tracker = MetricTracker()

    visualize = (epoch % 10 == 0)

    # 添加带进度条的迭代
    progress_bar = tqdm(loader,
                        desc=f"{COLOR['TRAIN']}Epoch {epoch}/{total_epochs} [Train]{COLOR['ENDC']}",
                        bar_format="{l_bar}{bar:20}{r_bar}",
                        leave=False)

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs_origin_images = inputs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        # 特征可视化（每epoch最后一个batch）
        if model_name not in ["Convnext","Swin-Transformer","Swin-Convnext","MedicalNet_Single_lca"]:
            if visualize and batch_idx == len(loader) - 1:
                with torch.no_grad():
                    # 获取特征图
                    heatmaps = model.feature_maps  # 包含三个阶段的特征图

                    # 生成可视化
                    plot_feature_heatmaps(
                        inputs_origin_images[:3],  # 取前3个样本
                        heatmaps,
                        path=f'result/test_result/{dataset_name}/{model_name}/heatmaps_epoch{epoch}.png'
                    )

        tracker.update(loss, preds, probs, labels)

        # 实时更新进度条显示
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{(preds == labels).float().mean().item():.4f}"
        })

    return tracker.compute()


def validate(model, loader, criterion, device, mode='Val'):
    model.eval()
    tracker = MetricTracker()

    # 验证/测试进度条
    with torch.no_grad():
        progress_bar = tqdm(loader,
                            desc=f"{COLOR[mode.upper()]}]{mode} Progress{COLOR['ENDC']}",
                            bar_format="{l_bar}{bar:20}{r_bar}",
                            leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            tracker.update(loss, preds, probs, labels)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(preds == labels).float().mean().item():.4f}"
            })

    return tracker.compute()

# 在utils.py中扩展
class AdaptiveHybridLoss(nn.Module):
    """结合Focal Loss和Class-Balanced Loss的动态损失"""
    def __init__(self, beta=0.999, gamma=2, mode='dynamic'):
        super().__init__()
        self.beta = beta  # 平衡参数
        self.gamma = gamma  # Focal参数
        self.class_weights = None
        self.mode = mode

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 动态计算有效样本数
        if self.mode == 'dynamic':
            hist = torch.histc(targets.float(), bins=inputs.size(1))
            freq = hist / hist.sum()
            self.class_weights = (1 - self.beta) / (1 - self.beta**freq)

        # Class-Balanced权重
        cb_weights = self.class_weights[targets]
        
        # Focal Loss组件
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 混合损失
        loss = (cb_weights * focal_loss).mean()
        return loss
    
    
class BalancedFocalLoss(nn.Module):
    """动态类别平衡的Focal Loss"""
    def __init__(self, alpha_mode='dynamic', gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha_mode = alpha_mode
        self.alpha = None

    def forward(self, inputs, targets):
        if self.alpha_mode == 'dynamic':
            self._update_alpha(targets, inputs.shape[1])
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt)**self.gamma * ce_loss
        return loss.mean()

    def _update_alpha(self, targets, n_classes):
        """动态调整类别权重"""
        hist = torch.histc(targets.float(), bins=n_classes, min=0, max=n_classes-1)
        freq = hist / hist.sum()
        self.alpha = (1.0 / (freq + 1e-5))  # 防止除零
        self.alpha = self.alpha / self.alpha.sum()

class DynamicWeightedCE(nn.Module):
    def __init__(self, base_weights, T=10):
        super().__init__()
        self.base_weights = base_weights  # 初始权重
        self.T = T  # 温度系数

    def forward(self, logits, labels):
        # 动态调整：根据当前epoch的预测难度
        probs = torch.softmax(logits.detach(), dim=1)
        pred_conf = probs[torch.arange(len(labels)), labels].mean()
        dynamic_scale = 1 + (1 - pred_conf) * self.T
        
        weights = self.base_weights * dynamic_scale
        loss = F.cross_entropy(logits, labels, weight=weights)
        return loss
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default="MedicalNet")

    args = parser.parse_args()
    
    normalized_path = os.path.normpath(args.data_dir)
    dataset_name = normalized_path.split("/")[-1]
    
    model_name = args.model_name
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalNet(num_classes=len(os.listdir(args.data_dir))).to(device)
   
    os.makedirs(f"result/test_result/{dataset_name}/{model_name}", exist_ok=True)
    
    # 打印模型信息
    print(f"{COLOR['HEADER']}\n{'=' * 40}")
    print(f"Training MedicalNet on {device}")
    print(f"Total epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"Classes: {os.listdir(args.data_dir)}")
    print("dataset_name:", dataset_name)
    print("model_name:", model_name)
    print(f"{'=' * 40}{COLOR['ENDC']}\n")

    # 混合优化器
    if model_name in ["Convnext", "Swin-Transformer", "Swin-Convnext"]:
        optimizer = AdamW([{'params': model.parameters(), 'lr': 1e-3, 'weight_decay': 0.02}])
    else:
        optimizer = AdamW([
            {'params': model.convnext.parameters(), 'lr': 1e-4, 'weight_decay': 0.03},
            {'params': model.swin.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
            {'params': model.fusions.parameters(), 'weight_decay': 0.001},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ])

    # 新增余弦衰减学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,  # 总衰减周期
        eta_min=1e-5,       # 最小学习率
        last_epoch=-1
    )
    
    # 新增类别统计
    class_names = sorted(os.listdir(args.data_dir))
    class_counts = [len(os.listdir(os.path.join(args.data_dir, name))) for name in class_names]
    

    criterion = nn.CrossEntropyLoss()
    # 数据加载
    train_loader, val_loader, test_loader = create_loaders(args.data_dir, args.batch_size)

    # 训练循环
    best_auc = 0
    history = {'train': {'loss': [], 'acc': []},
               'val': {'loss': [], 'acc': [], 'auc': []},
               'lr': []}

    # 主进度条
    epoch_bar = tqdm(range(args.epochs),
                     desc=f"{COLOR['HEADER']}Overall Progress{COLOR['ENDC']}",
                     bar_format="{l_bar}{bar:20}{r_bar}")
    
    os.makedirs(f"logs/{dataset_name}/{model_name}/", exist_ok=True)
    for epoch in epoch_bar:
        start_time = time.time()
        # 训练阶段
        train_res = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, args.epochs,dataset_name,model_name)

        # 验证阶段
        val_res = validate(model, val_loader, criterion, device, mode='Val')

        # 更新学习率（新增步骤）
        if epoch % 5 == 0:
            scheduler.step()

        # 记录当前学习率（新增）
        current_lr = optimizer.param_groups[0]['lr']  # 记录第一个参数组的学习率
        history['lr'].append(current_lr)

        # 记录历史数据
        for metric in history['train']:
            history['train'][metric].append(train_res[metric])
        for metric in history['val']:
            history['val'][metric].append(val_res[metric])

        # 保存最佳模型
        if val_res['auc'] > best_auc:
            best_auc = val_res['auc']
            os.makedirs(f"models/{dataset_name}/{model_name}", exist_ok=True)

            torch.save(model.state_dict(), f'models/{dataset_name}/{model_name}/best_model.pth')
            epoch_bar.set_postfix({'best_auc': f"{best_auc:.4f}"})

        # 打印epoch结果
        epoch_time = time.time() - start_time
        print(f"{COLOR['TRAIN']}[Train] Loss: {train_res['loss']:.4f} | Acc: {train_res['acc']:.4f}{COLOR['ENDC']}", end="   ")
        print(
            f"{COLOR['VAL']}[Val] Loss: {val_res['loss']:.4f} | Acc: {val_res['acc']:.4f} | AUC: {val_res['auc']:.4f}{COLOR['ENDC']}")
        print(f"Precision: {val_res['precision']:.4f} | Recall: {val_res['recall']:.4f}", end=" ")
        print(f"F1-score: {val_res['f1']:.4f} | Specificity: {val_res['specificity']:.4f}", end=" ")
        print(f"Epoch time: {epoch_time:.1f}s | Best AUC: {best_auc:.4f}\n")
        
        # 定义日志内容
        log_content = (
            f"{COLOR['TRAIN']}[EPOCH:{epoch} Train] Loss: {train_res['loss']:.4f} | Acc: {train_res['acc']:.4f}{COLOR['ENDC']}   \n"
            f"{COLOR['VAL']}[EPOCH:{epoch} Val] Loss: {val_res['loss']:.4f} | Acc: {val_res['acc']:.4f} | AUC: {val_res['auc']:.4f}{COLOR['ENDC']}\n"
            f"Precision: {val_res['precision']:.4f} | Recall: {val_res['recall']:.4f} "
            f"F1-score: {val_res['f1']:.4f} | Specificity: {val_res['specificity']:.4f} "
            f"Epoch time: {epoch_time:.1f}s | Best AUC: {best_auc:.4f}\n"
        )

        # 将日志内容写入 logs.txt 文件
        # 实时写入并刷新缓冲区
        log_path = f"logs/{dataset_name}/{model_name}/training_log.txt"
        with open(log_path, "a", buffering=1) as log_file:  # 行缓冲模式
            log_file.write(log_content.replace(COLOR['TRAIN'], '')
                          .replace(COLOR['VAL'], '')
                          .replace(COLOR['ENDC'], ''))  # 去除颜色代码
        if epoch % 5 == 0:
           # 可视化输出
            test_res = validate(model, test_loader, criterion, device, mode='Test')
            plot_confusion_matrix(test_res['targets'],test_res['preds'],class_names=os.listdir(args.data_dir),
                                  dataset_name=dataset_name,model_name=model_name,epoch=epoch)
            plot_roc_curve(test_res['targets'],test_res['probs'],class_names=os.listdir(args.data_dir),
                                dataset_name=dataset_name,model_name=model_name,epoch=epoch)

    # 最终测试
    model.load_state_dict(torch.load(f'models/{dataset_name}/{model_name}/best_model.pth'))
    test_res = validate(model, test_loader, criterion, device, mode='Test')
    plot_learning_curves(history['train'], history['val'],dataset_name,model_name)
    
    # 最终测试结果写入日志
    final_test_log = (
        f"\n{COLOR['TEST']}{'=' * 40}\n"
        "Final Results:\n"
        f"Accuracy: {test_res['acc']:.4f} | AUC: {test_res['auc']:.4f}\n"
        f"Precision: {test_res['precision']:.4f} | Recall: {test_res['recall']:.4f}\n"
        f"F1-score: {test_res['f1']:.4f} | Specificity: {test_res['specificity']:.4f}\n"
        f"{'=' * 40}{COLOR['ENDC']}\n\n"
    )
    print(final_test_log)
    # 写入最终测试结果
    with open(log_path, "a", buffering=1) as log_file:
        log_file.write(final_test_log.replace(COLOR['TEST'], '')
                      .replace(COLOR['ENDC'], ''))

    # 打印最终结果
    print(f"\n{COLOR['TEST']}{'=' * 40}")
    print("Final Test Results:")
    print(f"Accuracy: {test_res['acc']:.4f} | AUC: {test_res['auc']:.4f}")
    print(f"Precision: {test_res['precision']:.4f} | Recall: {test_res['recall']:.4f}")
    print(f"{'=' * 40}{COLOR['ENDC']}")

 
if __name__ == '__main__':

    main()
