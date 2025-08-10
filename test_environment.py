#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFloorplan 环境测试脚本
用于验证虚拟环境 dfp 中的所有依赖是否正确安装
"""

import sys
import os

def test_imports():
    """测试所有必要模块的导入"""
    print("=" * 60)
    print("DeepFloorplan 环境测试")
    print("=" * 60)
    
    # 测试基础包
    test_results = {}
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        test_results['tensorflow'] = True
    except Exception as e:
        print(f"❌ TensorFlow: {e}")
        test_results['tensorflow'] = False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        test_results['numpy'] = True
    except Exception as e:
        print(f"❌ NumPy: {e}")
        test_results['numpy'] = False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        test_results['opencv'] = True
    except Exception as e:
        print(f"❌ OpenCV: {e}")
        test_results['opencv'] = False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        test_results['matplotlib'] = True
    except Exception as e:
        print(f"❌ Matplotlib: {e}")
        test_results['matplotlib'] = False
    
    try:
        from PIL import Image
        print(f"✅ Pillow: {Image.__version__}")
        test_results['pillow'] = True
    except Exception as e:
        print(f"❌ Pillow: {e}")
        test_results['pillow'] = False
    
    try:
        import pytesseract
        print(f"✅ PyTesseract: {pytesseract.__version__}")
        test_results['pytesseract'] = True
    except Exception as e:
        print(f"❌ PyTesseract: {e}")
        test_results['pytesseract'] = False
    
    try:
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
        test_results['scipy'] = True
    except Exception as e:
        print(f"❌ SciPy: {e}")
        test_results['scipy'] = False
    
    print("\n" + "=" * 60)
    print("项目模块测试")
    print("=" * 60)
    
    # 测试项目模块
    try:
        import net
        print("✅ net.py 模块导入成功")
        test_results['net'] = True
    except Exception as e:
        print(f"❌ net.py 模块: {e}")
        test_results['net'] = False
    
    try:
        import preprocess
        print("✅ preprocess.py 模块导入成功")
        test_results['preprocess'] = True
    except Exception as e:
        print(f"❌ preprocess.py 模块: {e}")
        test_results['preprocess'] = False
    
    try:
        import postprocess
        print("✅ postprocess.py 模块导入成功")
        test_results['postprocess'] = True
    except Exception as e:
        print(f"❌ postprocess.py 模块: {e}")
        test_results['postprocess'] = False
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"总计测试: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！环境配置成功！")
        return True
    else:
        print(f"\n⚠️ 有 {total_tests - passed_tests} 个测试失败，请检查环境配置")
        return False

def check_demo_files():
    """检查演示文件是否存在"""
    print("\n" + "=" * 60)
    print("演示文件检查")
    print("=" * 60)
    
    demo_files = [
        "./demo/45719584.jpg",
        "./demo/45765448.jpg", 
        "./demo/demo.jpg"
    ]
    
    for file_path in demo_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (文件不存在)")

def main():
    """主函数"""
    print(f"Python 版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python 路径: {sys.executable}")
    
    # 运行测试
    success = test_imports()
    check_demo_files()
    
    print("\n" + "=" * 60)
    if success:
        print("环境测试完成！可以开始使用 DeepFloorplan 项目。")
        print("\n建议的下一步:")
        print("1. 运行: python demo.py --im_path=./demo/45719584.jpg")
        print("2. 如果需要训练: python main.py --phase=Train")
    else:
        print("环境测试发现问题，请检查依赖安装。")
    print("=" * 60)

if __name__ == "__main__":
    main()
