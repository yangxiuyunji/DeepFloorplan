#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的图像查看器，用于在VS Code中预览生成的九宫格图像
"""

import cv2
import numpy as np
from pathlib import Path

def show_image(image_path):
    """显示图像"""
    if not Path(image_path).exists():
        print(f"图像文件不存在: {image_path}")
        return
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"图像信息:")
    print(f"  路径: {image_path}")
    print(f"  尺寸: {image.shape[1]} x {image.shape[0]} 像素")
    print(f"  通道: {image.shape[2]}")
    
    # 调整图像大小以适合显示
    h, w = image.shape[:2]
    max_size = 1200
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"  调整后尺寸: {new_w} x {new_h}")
    
    # 显示图像
    window_name = f"九宫格可视化 - {Path(image_path).name}"
    cv2.imshow(window_name, image)
    
    print(f"\n图像已显示在窗口中，按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 显示刚生成的九宫格图像
    image_path = "output/demo4_new_result_edited_luoshu_grid.png"
    show_image(image_path)
