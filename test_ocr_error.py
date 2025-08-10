#!/usr/bin/env python3
"""测试OCR错误处理机制"""

import sys
import os
import numpy as np

# 临时禁用PaddleOCR来测试错误处理
sys.modules['paddleocr'] = None

# 导入我们的OCR模块并测试调用
try:
    from utils.ocr_enhanced import extract_room_text
    
    # 创建一个测试图像
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 尝试调用OCR函数（这时应该会抛出错误）
    result = extract_room_text(test_image)
    print("❌ 测试失败：应该抛出错误但没有")
    
except RuntimeError as e:
    print("✅ 测试成功：正确抛出了RuntimeError")
    print(f"错误信息：{e}")
except Exception as e:
    print(f"❌ 测试失败：抛出了错误类型：{type(e).__name__}: {e}")
