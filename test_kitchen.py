#!/usr/bin/env python3
"""
测试厨房识别功能的脚本
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from demo import main
import argparse

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def test_kitchen_detection():
    """测试厨房识别功能"""
    print("🏠 开始测试厨房识别功能...")
    
    # 测试图片列表
    test_images = [
        './demo/45765448.jpg',
        './demo/45719584.jpg', 
        './demo/47541863.jpg'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📍 测试图片: {img_path}")
            
            # 创建参数对象
            args = argparse.Namespace()
            args.im_path = img_path
            args.disable_closet = False
            
            try:
                # 运行识别
                main(args)
                print(f"✅ {img_path} 识别完成")
            except Exception as e:
                print(f"❌ {img_path} 识别失败: {e}")
        else:
            print(f"⚠️  图片不存在: {img_path}")

def print_kitchen_info():
    """打印厨房识别的技术信息"""
    print("=" * 60)
    print("🍳 厨房识别功能技术说明")
    print("=" * 60)
    print()
    print("🎯 识别方法:")
    print("1. OCR文字识别 - 检测图片中的'厨房'、'kitchen'等文字")
    print("2. 空间分析 - 分析房间的大小、形状等特征")
    print("3. 临近关系 - 考虑厨房通常与餐厅相邻的布局特点")
    print()
    print("🎨 颜色编码:")
    print("• 厨房: RGB(255, 200, 100) - 橙黄色")
    print("• 客厅/餐厅: RGB(224, 255, 192) - 浅绿色") 
    print()
    print("📝 支持的文字标识:")
    print("• 中文: 厨房、烹饪")
    print("• 英文: kitchen, cook")
    print()

if __name__ == "__main__":
    print_kitchen_info()
    test_kitchen_detection()
    print("\n🎉 厨房识别测试完成！")
