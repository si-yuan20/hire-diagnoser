##############################
# data_prepare.py (修正版)
##############################
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import torch
from torch.utils.data import Subset
from albumentations.pytorch import ToTensorV2

class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        classes = sorted(os.listdir(root_dir))
        self.file_paths = []
        self.labels = []

        for cls_idx, cls_name in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                self.file_paths.append(os.path.join(cls_dir, fname))
                self.labels.append(cls_idx)

        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = cv2.imread(self.file_paths[idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # 尝试读取图像
            # img = cv2.imread(img)
            if img is None:
                raise ValueError(f"OpenCV returned None for {img_path}")
        
            # 颜色空间转换
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            # ... 后续处理代码 ...
        
        except Exception as e:
            print(f"Error processing image: {img_path}")
            print(f"Error details: {str(e)}")
            # 可选：返回空数据或跳过该样本
            return None, None  # 需与数据集结构匹配

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']  # 这里已经是CxHxW的tensor

        return img, torch.tensor(self.labels[idx], dtype=torch.long)  # 确保label类型正确

def create_loaders(data_dir, batch_size=32):
    # 定义不同的transform组合
    train_transform = A.Compose([
        A.Resize(height=256, width=256),  # 先放大
        A.RandomResizedCrop(
            height=224,  # 明确使用height/width参数
            width=224,
            scale=(0.6, 1.0),
            ratio=(0.7, 1.3),  # 必须包含ratio参数
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),  # 保持参数格式统一
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 创建完整的未转换数据集
    full_dataset = MedicalDataset(data_dir, transform=None)

    # 划分索引
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=full_dataset.labels
    )
    test_idx, val_idx = train_test_split(
        test_idx,
        test_size=0.5,
        stratify=[full_dataset.labels[i] for i in test_idx]
    )

    # 创建带不同transform的子数据集
    train_dataset = MedicalDataset(data_dir, transform=train_transform)
    train_dataset.file_paths = [full_dataset.file_paths[i] for i in train_idx]
    train_dataset.labels = [full_dataset.labels[i] for i in train_idx]

    val_dataset = MedicalDataset(data_dir, transform=val_transform)
    val_dataset.file_paths = [full_dataset.file_paths[i] for i in val_idx]
    val_dataset.labels = [full_dataset.labels[i] for i in val_idx]

    test_dataset = MedicalDataset(data_dir, transform=val_transform)
    test_dataset.file_paths = [full_dataset.file_paths[i] for i in test_idx]
    test_dataset.labels = [full_dataset.labels[i] for i in test_idx]

    # 创建DataLoader
    return (
        DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size, num_workers=2, pin_memory=True),
        DataLoader(test_dataset, batch_size, num_workers=2, pin_memory=True)
    )