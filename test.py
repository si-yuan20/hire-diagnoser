# test_model_metrics_real.py
import torch
import time
import os
import argparse
import numpy as np
from model import MedicalNet
from data_prepare import create_loaders  # 使用真实的数据加载器
from torch.utils.data import DataLoader
import glob

def measure_fps_with_real_data(model, data_loader, device='cuda', num_batches=50):
    """使用真实数据测量模型FPS"""
    model.eval()
    model.to(device)
    
    # 预热
    warmup_batches = 10
    batch_count = 0
    
    with torch.no_grad():
        # 预热阶段
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= warmup_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
        
        # 正式测量
        start_time = time.time()
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            batch_count += 1
        
        # 同步GPU操作
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
    
    # 计算FPS：总图片数 / 总时间
    total_images = batch_count * data_loader.batch_size
    total_time = end_time - start_time
    fps = total_images / total_time if total_time > 0 else 0
    
    return fps

def calculate_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # 格式化输出
    if total_params > 1e6:
        params_str = f"{total_params/1e6:.2f}M"
    else:
        params_str = f"{total_params/1e3:.2f}K"
    
    return params_str, total_params

def load_trained_model(model_class, dataset_path, model_name, device):
    """加载训练好的模型权重"""
    # 获取数据集名称
    normalized_path = os.path.normpath(dataset_path)
    dataset_name = normalized_path.split("/")[-1]
    
    # 创建模型实例
    num_classes = len(os.listdir(dataset_path))
    model = model_class(num_classes=num_classes).to(device)
    
    # 查找模型权重文件
    weight_paths = [
        f"models/{dataset_name}/{model_name}/best_model.pth",
        f"models/{dataset_name}/{model_name}/latest_model.pth",
        f"checkpoints/{dataset_name}/{model_name}/best_model.pth"
    ]
    
    weight_loaded = False
    for weight_path in weight_paths:
        if os.path.exists(weight_path):
            try:
                model.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"✓ Loaded weights from {weight_path}")
                weight_loaded = True
                break
            except Exception as e:
                print(f"✗ Failed to load {weight_path}: {e}")
    
    if not weight_loaded:
        print(f"⚠ No pre-trained weights found for {model_name}, using random initialization")
    
    return model

def test_all_models_with_real_data(dataset_path, device='cuda', num_batches=50, batch_size=32):
    """使用真实数据测试所有模型的计算指标"""
    
    # 创建真实数据加载器
    _, _, test_loader = create_loaders(dataset_path, batch_size)
    
    # 创建数据子集用于测试（避免测试时间过长）
    real_test_loader = create_subset_loader(test_loader, num_batches * batch_size)
    
    models_config = {
        "MedicalNet": (MedicalNet, "MedicalNet"),
        "MedicalNet_SE": (MedicalNet_SE, "MedicalNet_SE"),
        "MedicalNet_CBAM": (MedicalNet_CBAM, "MedicalNet_CBAM"),
        "MedicalNet_ECA": (MedicalNet_ECA, "MedicalNet_ECA"),
        "ConvNeXt": (ConvNeXtClassifier, "Convnext"),
        "Swin-Transformer": (SwinClassifier, "Swin-Transformer"),
        "Swin-Convnext": (FusionClassifier, "Swin-Convnext"),
        "MedicalNet_Single_lca": (MedicalNet_Single_lca, "MedicalNet_Single_lca")
    }
    
    results = []
    
    print("=" * 60)
    print(f"{'Model Name':<20} {'Params':<12} {'FPS':<12} {'Status':<10}")
    print("=" * 60)
    
    for model_display_name, (model_class, model_weight_name) in models_config.items():
        try:
            print(f"Testing {model_display_name}...")
            
            # 加载训练好的模型
            model = load_trained_model(model_class, dataset_path, model_weight_name, device)
            
            # 计算参数量
            params_str, total_params = calculate_parameters(model)
            
            # 使用真实数据测量FPS
            fps = measure_fps_with_real_data(
                model, 
                real_test_loader, 
                device=device, 
                num_batches=num_batches
            )
            
            status = "✓ Trained" if any([
                os.path.exists(f"models/{os.path.basename(dataset_path)}/{model_weight_name}/best_model.pth"),
                os.path.exists(f"models/{os.path.basename(dataset_path)}/{model_weight_name}/latest_model.pth")
            ]) else "○ Random"
            
            results.append({
                'model_name': model_display_name,
                'params': params_str,
                'fps': round(fps, 2),
                'total_params': total_params,
                'status': status
            })
            
            print(f"{model_display_name:<20} {params_str:<12} {fps:<12.2f} {status:<10}")
            
        except Exception as e:
            print(f"Error testing {model_display_name}: {str(e)}")
            results.append({
                'model_name': model_display_name,
                'params': 'N/A',
                'fps': 'N/A',
                'total_params': 0,
                'status': '✗ Failed'
            })
    
    return results

def create_subset_loader(original_loader, max_samples):
    """创建数据子集加载器"""
    dataset = original_loader.dataset
    indices = torch.randperm(len(dataset))[:max_samples]
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    
    return DataLoader(
        subset,
        batch_size=original_loader.batch_size,
        shuffle=False,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory
    )

def save_results_to_file(results, dataset_name, filename=None):
    """将结果保存到文件"""
    if filename is None:
        filename = f"model_metrics_real_{dataset_name}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Real Data Model Metrics Test Results - {dataset_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Model Name':<20} {'Params':<12} {'FPS':<12} {'Status':<10}\n")
        f.write("=" * 70 + "\n")
        
        for result in results:
            f.write(f"{result['model_name']:<20} {result['params']:<12} {result['fps']:<12} {result['status']:<10}\n")
        
        # 添加详细统计
        f.write("\n" + "=" * 50 + "\n")
        f.write("DETAILED ANALYSIS\n")
        f.write("=" * 50 + "\n")
        
        valid_results = [r for r in results if r['fps'] != 'N/A']
        if valid_results:
            fastest = max(valid_results, key=lambda x: x['fps'])
            smallest = min(valid_results, key=lambda x: x['total_params'])
            
            f.write(f"Fastest Model: {fastest['model_name']} ({fastest['fps']} FPS)\n")
            f.write(f"Smallest Model: {smallest['model_name']} ({smallest['params']} parameters)\n")
            
            # 计算效率评分
            f.write("\nEfficiency Ranking (FPS per Million Parameters):\n")
            efficiency_scores = []
            for result in valid_results:
                if result['total_params'] > 0:
                    efficiency = result['fps'] / (result['total_params'] / 1e6)
                    efficiency_scores.append((result['model_name'], efficiency))
            
            efficiency_scores.sort(key=lambda x: x[1], reverse=True)
            for i, (name, score) in enumerate(efficiency_scores, 1):
                f.write(f"  {i:2d}. {name:<20} {score:.2f} FPS/Mparam\n")
    
    print(f"\nResults saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Test model FPS and parameters with real data')
    parser.add_argument('--data_dir', required=True, help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for testing')
    parser.add_argument('--num_batches', type=int, default=50,
                       help='Number of batches for FPS measurement')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        args.device = 'cpu'
    
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    
    print(f"Testing models with real data from: {args.data_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Total test images: {args.num_batches * args.batch_size}")
    print()
    
    # 测试所有模型
    results = test_all_models_with_real_data(
        dataset_path=args.data_dir,
        device=args.device,
        num_batches=args.num_batches,
        batch_size=args.batch_size
    )
    
    # 保存结果到文件
    save_results_to_file(results, dataset_name)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    valid_results = [r for r in results if r['fps'] != 'N/A']
    if valid_results:
        fastest_model = max(valid_results, key=lambda x: x['fps'])
        smallest_model = min(valid_results, key=lambda x: x['total_params'])
        
        print(f"🏆 Fastest Model: {fastest_model['model_name']} ({fastest_model['fps']} FPS)")
        print(f"📦 Smallest Model: {smallest_model['model_name']} ({smallest_model['params']} parameters)")
        
        # 效率排名
        print("\n🏅 Efficiency Ranking (Higher is better):")
        efficiency_scores = []
        for result in valid_results:
            if result['total_params'] > 0:
                efficiency = result['fps'] / (result['total_params'] / 1e6)
                efficiency_scores.append((result['model_name'], efficiency, result['status']))
        
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (name, score, status) in enumerate(efficiency_scores, 1):
            print(f"  {i:2d}. {name:<20} {score:.2f} FPS/Mparam [{status}]")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
