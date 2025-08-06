#!/usr/bin/env python3
"""
快速测试TensorFlow导入和基础功能 - 避免Unicode问题
"""
import os
import sys

# 设置控制台编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("开始测试TensorFlow...")

try:
    import numpy as np
    print(f"[OK] NumPy {np.__version__} 导入成功")
except Exception as e:
    print(f"[ERROR] NumPy导入失败: {e}")
    sys.exit(1)

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print(f"[OK] TensorFlow {tf.__version__} v1兼容模式导入成功")
except Exception as e:
    print(f"[ERROR] TensorFlow导入失败: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("[OK] PIL导入成功")
except Exception as e:
    print(f"[ERROR] PIL导入失败: {e}")
    sys.exit(1)

print("\n测试基础TensorFlow操作...")
try:
    # 创建简单计算图
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = tf.add(a, b)
    
    with tf.Session() as sess:
        result = sess.run(c)
        print(f"[OK] TensorFlow计算测试成功: 2 + 3 = {result}")
except Exception as e:
    print(f"[ERROR] TensorFlow计算测试失败: {e}")
    sys.exit(1)

print("\n测试图像读取...")
try:
    img = Image.open("./demo/45719584.jpg")
    img_array = np.array(img)
    print(f"[OK] 图像读取成功: {img_array.shape}")
except Exception as e:
    print(f"[ERROR] 图像读取失败: {e}")

print("\n测试预训练模型文件...")
try:
    meta_file = "./pretrained/pretrained_r3d.meta"
    if os.path.exists(meta_file):
        # 尝试加载meta文件
        saver = tf.train.import_meta_graph(meta_file)
        print("[OK] 预训练模型meta文件加载成功")
        
        # 检查模型文件
        checkpoint_file = "./pretrained/pretrained_r3d"
        if os.path.exists(checkpoint_file + ".index"):
            print("[OK] 预训练模型checkpoint文件存在")
        else:
            print("[ERROR] 预训练模型checkpoint文件不存在")
    else:
        print("[ERROR] 预训练模型meta文件不存在")
except Exception as e:
    print(f"[ERROR] 预训练模型加载失败: {e}")

print("\n所有测试完成！")
print("如果上述测试都通过，那么demo应该可以正常运行。")
print("现在尝试运行demo...")

# 尝试运行demo的核心部分
try:
    print("\n=== 开始demo测试 ===")
    
    # 加载图像
    from PIL import Image
    import numpy as np
    
    def imresize(img, size):
        """Resize image using PIL"""
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        if len(img.shape) == 3:
            h, w, c = size if len(size) == 3 else (*size, img.shape[2])
            img_pil = Image.fromarray(img)
            img_resized = img_pil.resize((w, h))
            return np.array(img_resized)
        else:
            h, w = size
            img_pil = Image.fromarray(img)
            img_resized = img_pil.resize((w, h))
            return np.array(img_resized)
    
    # 读取并处理图像
    img_path = "./demo/45719584.jpg"
    img = Image.open(img_path)
    img = np.array(img)
    print(f"[OK] 原始图像形状: {img.shape}")
    
    # 缩放图像
    img_resized = imresize(img, (512, 512))
    print(f"[OK] 缩放后图像形状: {img_resized.shape}")
    
    # 归一化
    img_normalized = img_resized.astype(np.float32) / 255.0
    print(f"[OK] 归一化完成: {img_normalized.shape}, 范围: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    
    print("[OK] 图像预处理测试成功！")
    
except Exception as e:
    print(f"[ERROR] Demo测试失败: {e}")
    import traceback
    traceback.print_exc()
