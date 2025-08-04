#!/usr/bin/env python3
"""
快速测试TensorFlow导入和基础功能
"""
print("开始测试TensorFlow...")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} 导入成功")
except Exception as e:
    print(f"✗ NumPy导入失败: {e}")
    exit(1)

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print(f"✓ TensorFlow {tf.__version__} v1兼容模式导入成功")
except Exception as e:
    print(f"✗ TensorFlow导入失败: {e}")
    exit(1)

try:
    from PIL import Image
    print("✓ PIL导入成功")
except Exception as e:
    print(f"✗ PIL导入失败: {e}")
    exit(1)

print("\n测试基础TensorFlow操作...")
try:
    # 创建简单计算图
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = tf.add(a, b)
    
    with tf.Session() as sess:
        result = sess.run(c)
        print(f"✓ TensorFlow计算测试成功: 2 + 3 = {result}")
except Exception as e:
    print(f"✗ TensorFlow计算测试失败: {e}")
    exit(1)

print("\n测试图像读取...")
try:
    img = Image.open("./demo/45719584.jpg")
    img_array = np.array(img)
    print(f"✓ 图像读取成功: {img_array.shape}")
except Exception as e:
    print(f"✗ 图像读取失败: {e}")

print("\n测试预训练模型文件...")
try:
    import os
    meta_file = "./pretrained/pretrained_r3d.meta"
    if os.path.exists(meta_file):
        # 尝试加载meta文件
        saver = tf.train.import_meta_graph(meta_file)
        print("✓ 预训练模型meta文件加载成功")
        
        # 检查模型文件
        checkpoint_file = "./pretrained/pretrained_r3d"
        if os.path.exists(checkpoint_file + ".index"):
            print("✓ 预训练模型checkpoint文件存在")
        else:
            print("✗ 预训练模型checkpoint文件不存在")
    else:
        print("✗ 预训练模型meta文件不存在")
except Exception as e:
    print(f"✗ 预训练模型加载失败: {e}")

print("\n所有测试完成！")
print("如果上述测试都通过，那么demo应该可以正常运行。")
