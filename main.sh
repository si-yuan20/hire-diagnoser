# #!/bin/bash

# # 设置环境变量（如果需要）
# # export CUDA_VISIBLE_DEVICES=0  # 如果需要指定 GPU


# echo "Starting First experiment with Brain dataset..."
# python main.py --data_dir /root/autodl-fs/Brain/Brain/ --batch_size 64 --epochs 50 --model_name "Convnext"
# python main.py --data_dir /root/autodl-tmp/Brain/ --batch_size 74 --epochs 50 --model_name "MedicalNet"
# python main.py --data_dir /root/autodl-tmp/Brain/ --batch_size 74 --epochs 50 --model_name "MedicalNet_SE"
# python main.py --data_dir /root/autodl-tmp/Brain/ --batch_size 74 --epochs 50 --model_name "MedicalNet_CBAM"
# python main.py --data_dir /root/autodl-tmp/Brain/ --batch_size 74 --epochs 50 --model_name "MedicalNet_ECA"
# python main.py --data_dir /root/autodl-fs/Brain/Brain/ --batch_size 64 --epochs 50 --model_name "Swin-Transformer"
# python main.py --data_dir /root/autodl-fs/Brain/Brain/ --batch_size 64 --epochs 50 --model_name "Swin-Convnext"
# python main.py --data_dir /root/autodl-fs/Brain/Brain/ --batch_size 64 --epochs 50 --model_name "MedicalNet_Single_lca"


# # 第一个实验
# echo "Starting Second experiment with Raabin-WBC dataset......"
# python main.py --data_dir /root/autodl-fs/Raabin-WBC/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "Convnext"
# python main.py --data_dir /root/autodl-tmp/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "MedicalNet"
# python main.py --data_dir /root/autodl-tmp/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "MedicalNet_SE"
# python main.py --data_dir /root/autodl-tmp/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "MedicalNet_CBAM"
# python main.py --data_dir /root/autodl-tmp/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "MedicalNet_ECA"
# python main.py --data_dir /root/autodl-fs/Raabin-WBC/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "Swin-Transformer"
# python main.py --data_dir /root/autodl-fs/Raabin-WBC/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "Swin-Convnext"
# python main.py --data_dir /root/autodl-fs/Raabin-WBC/Raabin-WBC/ --batch_size 64 --epochs 50 --model_name "MedicalNet_Single_lca"

# echo "First experiment completed."

# 第二个实验
echo "Starting Third experiment with LC25000 dataset..."
# python main.py --data_dir /root/autodl-tmp/LC25000/ --batch_size 64 --epochs 50 --model_name "MedicalNet"
# python main.py --data_dir /root/autodl-tmp/LC25000/ --batch_size 64 --epochs 50 --model_name "MedicalNet_SE"
# python main.py --data_dir /root/autodl-tmp/LC25000/ --batch_size 64 --epochs 50 --model_name "MedicalNet_CBAM"
# python main.py --data_dir /root/autodl-tmp/LC25000/ --batch_size 64 --epochs 50 --model_name "MedicalNet_ECA"
# python main.py --data_dir /root/autodl-fs/LC25000/ --batch_size 64 --epochs 50 --model_name "Convnext"
# python main.py --data_dir /root/autodl-fs/LC25000/ --batch_size 64 --epochs 50 --model_name "Swin-Transformer"
python main.py --data_dir /root/autodl-fs/LC25000/ --batch_size 64 --epochs 50 --model_name "Swin-Convnext"
python main.py --data_dir /root/autodl-fs/LC25000/ --batch_size 64 --epochs 50 --model_name "MedicalNet_Single_lca"

# echo "Second experiment completed."


# # 第四个实验
echo "Starting Four experiment with RetinalOCT dataset..."

python main.py --data_dir /root/autodl-fs/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "Convnext"
python main.py --data_dir /root/autodl-fs/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "Swin-Transformer"
python main.py --data_dir /root/autodl-fs/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "Swin-Convnext"
python main.py --data_dir /root/autodl-fs/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "MedicalNet_Single_lca"
# python main.py --data_dir /root/autodl-tmp/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "MedicalNet"
# python main.py --data_dir /root/autodl-tmp/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "MedicalNet_SE"
# python main.py --data_dir /root/autodl-tmp/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "MedicalNet_CBAM"
# python main.py --data_dir /root/autodl-tmp/RetinalOCT/ --batch_size 64 --epochs 50 --model_name "MedicalNet_ECA"

# echo "four experiment completed."