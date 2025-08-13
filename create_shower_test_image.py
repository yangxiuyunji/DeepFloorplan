#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建包含淋浴间文字的测试图像
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def create_test_floorplan_with_shower():
    """创建包含淋浴间标识的测试户型图"""
    
    # 创建基础图像 (白底)
    width, height = 600, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 绘制房间轮廓 (黑色线条)
    # 外墙
    cv2.rectangle(img, (50, 50), (550, 350), (0, 0, 0), 3)
    
    # 内部分隔线
    cv2.line(img, (200, 50), (200, 200), (0, 0, 0), 2)  # 竖线1
    cv2.line(img, (350, 50), (350, 350), (0, 0, 0), 2)  # 竖线2
    cv2.line(img, (50, 200), (350, 200), (0, 0, 0), 2)   # 横线1
    cv2.line(img, (200, 250), (550, 250), (0, 0, 0), 2)  # 横线2
    cv2.line(img, (450, 50), (450, 250), (0, 0, 0), 2)   # 竖线3（分隔淋浴间）
    
    # 门的标识 (细线)
    cv2.line(img, (175, 50), (175, 80), (128, 128, 128), 2)
    cv2.line(img, (325, 200), (325, 230), (128, 128, 128), 2)
    cv2.line(img, (425, 50), (425, 80), (128, 128, 128), 2)
    
    # 转换为PIL图像以添加中文文字
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试使用中文字体，如果没有则使用默认字体
    try:
        # Windows系统的中文字体
        font_large = ImageFont.truetype("msyh.ttc", 22)  # 微软雅黑
        font_medium = ImageFont.truetype("msyh.ttc", 18)
        font_small = ImageFont.truetype("msyh.ttc", 16)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 22)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # 添加房间标识文字
    draw.text((110, 120), "卫生间", fill=(0, 0, 0), font=font_medium)     # 左上房间
    draw.text((250, 120), "卧室", fill=(0, 0, 0), font=font_medium)      # 中上房间
    draw.text((380, 120), "淋浴间", fill=(0, 0, 0), font=font_medium)     # 右上房间（淋浴间）
    draw.text((480, 120), "主卧", fill=(0, 0, 0), font=font_medium)      # 最右房间
    draw.text((110, 280), "书房", fill=(0, 0, 0), font=font_medium)      # 左下房间
    draw.text((430, 200), "厨房", fill=(0, 0, 0), font=font_medium)      # 右下房间上
    draw.text((430, 320), "客厅", fill=(0, 0, 0), font=font_medium)      # 右下房间下
    
    # 添加英文和其他卫生间变体
    draw.text((250, 280), "shower", fill=(128, 128, 128), font=font_small)  # 中下添加英文shower
    draw.text((80, 180), "盥洗室", fill=(64, 64, 64), font=font_small)      # 小字标注
    
    # 转换回OpenCV格式
    img_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_final

def main():
    print("🏗️ 创建包含淋浴间的测试户型图...")
    
    # 生成测试图像
    test_img = create_test_floorplan_with_shower()
    
    # 保存图像
    output_path = "demo/test_shower_floorplan.jpg"
    cv2.imwrite(output_path, test_img)
    
    print(f"✅ 测试图像已保存到: {output_path}")
    print("📋 图像包含以下房间标识:")
    print("   🚿 卫生间 (传统)")
    print("   🚿 淋浴间 (新增支持)")
    print("   🚿 shower (英文)")
    print("   🚿 盥洗室 (正式称谓)")
    print("   🏠 卧室, 主卧, 客厅, 厨房, 书房")
    
    # 显示图像信息
    height, width = test_img.shape[:2]
    print(f"📐 图像尺寸: {width}x{height}")
    
    return output_path

if __name__ == "__main__":
    test_path = main()
    print(f"\n🚀 可以运行以下命令测试淋浴间识别:")
    print(f"python demo_refactored_clean.py {test_path}")
