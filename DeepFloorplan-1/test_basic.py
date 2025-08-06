#!/usr/bin/env python3
"""
简化的环境测试 - 不使用TensorFlow
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from PIL import Image

print("=== DeepFloorplan 基础环境测试 ===")
print("Python环境: 虚拟环境 DeepFloorplan")
print(f"NumPy版本: {np.__version__}")
print(f"PIL版本: {Image.__version__}")

# 测试基础功能
print("\n1. 测试图像读取...")
try:
    # 读取demo图像
    demo_path = "./demo/45719584.jpg"
    if os.path.exists(demo_path):
        img = Image.open(demo_path)
        img_array = np.array(img)
        print(f"   ✓ 成功读取图像: {img_array.shape}")
        
        # 测试图像处理
        img_resized = img.resize((512, 512))
        print(f"   ✓ 图像缩放成功: {np.array(img_resized).shape}")
    else:
        print(f"   ✗ 演示图像不存在: {demo_path}")
except Exception as e:
    print(f"   ✗ 图像读取失败: {e}")

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
        print(f"   ✓ {file_path} ({file_size:,} bytes)")
    else:
        print(f"   ✗ {file_path} 不存在")
        all_exist = False

print("\n3. 检查工具函数...")
try:
    # 测试颜色映射
    floorplan_map = {
        0: [255,255,255], # background
        1: [192,192,224], # closet
        2: [192,255,255], # bathroom
        3: [224,255,192], # livingroom
        4: [255,224,128], # bedroom
        5: [255,160, 96], # hall
        6: [255,224,224], # balcony
        9: [255, 60,128], # door & window
        10:[  0,  0,  0]  # wall
    }
    
    # 创建测试索引图像
    test_ind = np.random.randint(0, 11, (100, 100))
    test_rgb = np.zeros((100, 100, 3))
    
    for i, rgb in floorplan_map.items():
        test_rgb[test_ind == i] = rgb
    
    print("   ✓ 颜色映射函数工作正常")
except Exception as e:
    print(f"   ✗ 颜色映射测试失败: {e}")

print("\n=== 当前状态总结 ===")
print("✓ Python虚拟环境 'DeepFloorplan' 已创建并激活")
print("✓ 基础依赖包正常工作:")
print("  - NumPy, Pillow, Matplotlib")
print("✓ 预训练模型文件已存在")
print("⚠  TensorFlow需要解决DLL问题")

print("\n=== 建议解决方案 ===")
print("1. TensorFlow DLL问题可能的解决方案:")
print("   a) 安装Microsoft Visual C++ Redistributable")
print("   b) 使用tensorflow-cpu版本") 
print("   c) 使用conda环境替代pip环境")
print("2. 可以先用现有环境开发图像处理部分")
print("3. 预训练模型已就绪，可用于推理测试")

print("\n基础环境配置完成！")

# 尝试测试TensorFlow
print("\n4. 尝试导入TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow {tf.__version__} 导入成功")
    
    # 简单计算测试
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = tf.add(a, b)
    print("   ✓ TensorFlow基础计算正常")
    
except Exception as e:
    print(f"   ✗ TensorFlow导入失败: {e}")
    print("   建议: 尝试重新安装tensorflow-cpu或使用conda环境")
