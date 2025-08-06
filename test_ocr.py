#!/usr/bin/env python3
"""测试OCR模块的基本功能"""

try:
    from utils.ocr import TEXT_LABEL_MAP, text_to_label
    print("✅ OCR模块导入成功")
    
    # 测试文本到标签的映射
    test_texts = ['卧室', 'bedroom', '客厅', 'living', '厨房', 'kitchen', '卫生间', 'bathroom']
    print("\n📋 测试房间类型识别：")
    for text in test_texts:
        label = text_to_label(text)
        print(f"  {text} -> 标签: {label}")
    
    print(f"\n🏠 支持的房间类型总数: {len(TEXT_LABEL_MAP)}")
    print("✅ OCR基础功能测试通过")
    
except ImportError as e:
    print(f"❌ OCR模块导入失败: {e}")
except Exception as e:
    print(f"❌ 测试过程中出现错误: {e}")

# 测试OCR功能是否可用（不实际调用OCR）
try:
    from utils.ocr import extract_room_text, fuse_ocr_and_segmentation
    
    # 测试空图像（模拟OCR不可用的情况）
    result = extract_room_text(None)
    print(f"📸 OCR提取测试（无输入）: {result}")
    
    print("✅ OCR函数调用测试通过")
    
except Exception as e:
    print(f"❌ OCR函数测试失败: {e}")
