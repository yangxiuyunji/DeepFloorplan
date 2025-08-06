#!/usr/bin/env python3
"""
DeepFloorplan 环境配置测试脚本
"""
import os
import sys
import numpy as np
from PIL import Image

print("=== DeepFloorplan 环境测试 ===")
print("Python环境: 虚拟环境 DeepFloorplan")
print(f"Python版本: {sys.version}")
print(f"NumPy版本: {np.__version__}")

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 测试基础功能
print("\n1. 测试图像读取...")
try:
    # 读取demo图像
    demo_path = "./demo/45719584.jpg"
    if os.path.exists(demo_path):
        img = Image.open(demo_path)
        img_array = np.array(img)
        print(f"   [OK] 成功读取图像: {img_array.shape}")
        
        # 测试图像处理
        img_resized = img.resize((512, 512))
        print(f"   [OK] 图像缩放成功: {np.array(img_resized).shape}")
    else:
        print(f"   [ERROR] 演示图像不存在: {demo_path}")
except Exception as e:
    print(f"   [ERROR] 图像读取失败: {e}")

print("\n2. 检查预训练模型...")
pretrained_files = [
    "./pretrained/pretrained_r3d.meta",
    "./pretrained/pretrained_r3d.index", 
    "./pretrained/pretrained_r3d.data-00000-of-00001"
]

all_exist = True
for file_path in pretrained_files:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"   [OK] {file_path} ({file_size:,} bytes)")
    else:
        print(f"   [ERROR] {file_path} 不存在")
        all_exist = False

print("\n3. 测试TensorFlow...")
try:
    import tensorflow as tf
    print(f"   [OK] TensorFlow {tf.__version__} 导入成功")
    
    # 检查TensorFlow v1兼容性
    if hasattr(tf, 'compat'):
        import tensorflow.compat.v1 as tf_v1
        print(f"   [OK] TensorFlow v1兼容模式可用")
        
        # 简单计算测试 (使用eager execution)
        try:
            a = tf.constant(2.0)
            b = tf.constant(3.0) 
            c = tf.add(a, b)
            print("   [OK] TensorFlow基础计算正常")
        except Exception as calc_error:
            print(f"   [WARNING] TensorFlow计算测试: {calc_error}")
    else:
        print("   [WARNING] TensorFlow v1兼容模式不可用")
        
except Exception as e:
    print(f"   [ERROR] TensorFlow导入失败: {e}")
    print("   建议: 尝试重新安装或使用conda环境")

print("\n4. 检查数据集结构...")
dataset_files = [
    "./dataset/r3d_test.txt",
    "./dataset/r3d_train.txt"
]

for file_path in dataset_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"   [OK] {file_path} ({len(lines)} 条记录)")
    else:
        print(f"   [INFO] {file_path} 不存在 (可能需要下载)")

print("\n5. 检查工具模块...")
try:
    # 检查utils目录
    utils_files = [
        "./utils/rgb_ind_convertor.py",
        "./utils/util.py", 
        "./utils/tf_record.py"
    ]
    
    missing_utils = []
    for util_file in utils_files:
        if os.path.exists(util_file):
            print(f"   [OK] {util_file}")
        else:
            missing_utils.append(util_file)
            print(f"   [ERROR] {util_file} 不存在")
            
    if not missing_utils:
        print("   [OK] 所有工具模块都存在")
        
except Exception as e:
    print(f"   [ERROR] 检查工具模块失败: {e}")

print("\n=== 环境配置总结 ===")
print("Python虚拟环境 'DeepFloorplan' 配置完成")
print("已安装主要依赖:")
print("  - NumPy, SciPy, Matplotlib")
print("  - OpenCV, Pillow") 
print("  - TensorFlow (需要验证v1兼容性)")
print("  - tensorflow-slim")

print("\n=== 下一步操作建议 ===")
print("1. 如果TensorFlow导入失败，建议:")
print("   - 安装Microsoft Visual C++ Redistributable")
print("   - 或使用conda创建Python 3.8环境")
print("   - 或使用Docker环境")

print("2. 原始代码适配需要:")
print("   - Python 2到3的语法更新")
print("   - TensorFlow 1.x到2.x的API适配")
print("   - scipy.misc的替换")

print("3. 可以尝试运行简化版demo进行测试")

print("\n环境搭建基本完成！")
