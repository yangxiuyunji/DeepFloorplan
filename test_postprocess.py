#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试后处理功能的简单脚本
"""

import os
import sys
import numpy as np
from PIL import Image
import tempfile
import shutil

def create_test_image():
    """创建一个简单的测试图像"""
    # 创建一个简单的彩色测试图像
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 添加一些颜色区域模拟房间
    img[50:100, 50:100] = [255, 0, 0]    # 红色区域
    img[100:150, 50:100] = [0, 255, 0]   # 绿色区域
    img[50:100, 100:150] = [0, 0, 255]   # 蓝色区域
    img[100:150, 100:150] = [255, 255, 0] # 黄色区域
    
    return img

def test_postprocess():
    """测试后处理功能"""
    print("🧪 开始测试后处理功能...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试图像
        test_img = create_test_image()
        test_path = os.path.join(temp_dir, "test_image.png")
        Image.fromarray(test_img).save(test_path)
        
        print(f"📝 创建测试图像: {test_path}")
        
        # 测试导入
        try:
            sys.path.append('./utils/')
            from rgb_ind_convertor import floorplan_fuse_map, rgb2ind, ind2rgb
            from util import fill_break_line, flood_fill, refine_room_region
            print("✅ 成功导入所有依赖模块")
        except ImportError as e:
            print(f"❌ 导入模块失败: {e}")
            return False
        
        # 测试基本函数
        try:
            # 测试图像读取
            from postprocess import imread, imsave
            img = imread(test_path)
            if img is not None:
                print("✅ 图像读取功能正常")
            else:
                print("❌ 图像读取失败")
                return False
                
            # 测试图像保存
            output_path = os.path.join(temp_dir, "output_test.png")
            if imsave(output_path, img):
                print("✅ 图像保存功能正常")
            else:
                print("❌ 图像保存失败")
                return False
                
        except Exception as e:
            print(f"❌ 基本功能测试失败: {e}")
            return False
    
    print("🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_postprocess()
    sys.exit(0 if success else 1)
