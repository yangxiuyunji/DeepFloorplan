#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试颜色显示问题
"""

from PIL import Image, ImageDraw

def test_actual_colors():
    """直接测试实际的颜色显示"""
    # 创建测试图像
    width, height = 600, 300
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # 透明度20%
    alpha = int(255 * 0.2)
    
    # 测试"吉"星颜色（红色）
    jixing_color = (255, 0, 0, alpha)    # 红色
    bbox1 = [50, 50, 200, 200]
    draw.pieslice(bbox1, 0, 90, fill=jixing_color, outline=(0, 0, 0, 255), width=2)
    draw.text((210, 100), f"吉星颜色 RGB{jixing_color}", fill=(0, 0, 0, 255))
    
    # 测试"凶"星颜色（黄色）
    xiongxing_color = (255, 255, 0, alpha)  # 黄色
    bbox2 = [300, 50, 450, 200]
    draw.pieslice(bbox2, 0, 90, fill=xiongxing_color, outline=(0, 0, 0, 255), width=2)
    draw.text((460, 100), f"凶星颜色 RGB{xiongxing_color}", fill=(0, 0, 0, 255))
    
    # 添加纯色参考（不透明）
    draw.rectangle([50, 220, 100, 270], fill=(255, 0, 0, 255))
    draw.text((110, 240), "纯红色参考", fill=(0, 0, 0, 255))
    
    draw.rectangle([300, 220, 350, 270], fill=(255, 255, 0, 255))
    draw.text((360, 240), "纯黄色参考", fill=(0, 0, 0, 255))
    
    # 保存测试图像
    output_path = "output/color_comparison_test.png"
    image.save(output_path)
    print(f"颜色对比测试图像已保存至: {output_path}")
    
    # 同时测试是否存在BGR混乱
    print("\n=== 颜色值检查 ===")
    print(f"吉星(红色): RGB{jixing_color}")
    print(f"凶星(黄色): RGB{xiongxing_color}")
    
    # 如果BGR被错误解释为RGB，会是什么样？
    print("\n=== 如果存在BGR/RGB混乱 ===")
    # (255, 0, 0) 在BGR中会被解释为蓝色
    # (255, 255, 0) 在BGR中会被解释为青色
    print("如果(255,0,0)被当作BGR解释: 会显示蓝色")
    print("如果(255,255,0)被当作BGR解释: 会显示青色")
    print("但这都不符合您看到的黄色情况...")

if __name__ == "__main__":
    test_actual_colors()
