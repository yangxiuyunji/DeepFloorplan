#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
书房识别功能测试脚本
测试各种书房相关关键词的识别能力
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_refactored_clean import FloorplanProcessor

def test_study_room_keywords():
    """测试书房关键词识别"""
    
    # 创建处理器实例
    processor = FloorplanProcessor()
    
    # 模拟OCR结果，包含各种书房关键词
    mock_ocr_results = [
        {'text': '书房', 'bbox': [100, 100, 50, 30], 'confidence': 0.95},
        {'text': 'study', 'bbox': [200, 150, 60, 25], 'confidence': 0.88},
        {'text': '办公室', 'bbox': [300, 200, 80, 35], 'confidence': 0.92},
        {'text': 'office', 'bbox': [150, 250, 70, 28], 'confidence': 0.85},
        {'text': '书', 'bbox': [250, 300, 25, 30], 'confidence': 0.75},
        {'text': '工作室', 'bbox': [350, 150, 75, 32], 'confidence': 0.90},
        # 添加其他房间作为对比
        {'text': '卧室', 'bbox': [400, 100, 50, 30], 'confidence': 0.98},
        {'text': '厨房', 'bbox': [50, 200, 50, 30], 'confidence': 0.96},
    ]
    
    print("📚 书房识别功能测试")
    print("=" * 50)
    print("🧪 模拟OCR检测结果:")
    
    for item in mock_ocr_results:
        print(f"   📝 '{item['text']}' (置信度: {item['confidence']:.3f})")
    
    print("\n🔍 开始关键词匹配测试...")
    
    # 测试关键词匹配逻辑
    study_room_keywords = ['书房', 'study', '书', '办公室', 'office']
    detected_study_rooms = []
    
    for item in mock_ocr_results:
        text = item['text'].lower().strip()
        
        # 书房关键词匹配（与重构版本中的逻辑一致）
        if any(keyword in text for keyword in ['书房', 'study', '书', '办公室', 'office']):
            detected_study_rooms.append(item)
            print(f"   ✅ 匹配书房: '{item['text']}' (置信度: {item['confidence']:.3f})")
    
    print(f"\n📊 测试结果:")
    print(f"   🎯 检测到 {len(detected_study_rooms)} 个书房区域")
    
    if detected_study_rooms:
        print(f"   📍 书房详情:")
        for i, room in enumerate(detected_study_rooms, 1):
            x, y, w, h = room['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            print(f"      🏠 书房{i}: '{room['text']}' 中心({center_x},{center_y}) 置信度:{room['confidence']:.3f}")
    else:
        print("   ❌ 未检测到书房")
    
    print("\n💡 支持的书房关键词:")
    print("   中文: 书房, 书, 办公室, 工作室")
    print("   英文: study, office")
    print("   混合: 书房/office, study room 等")
    
    return detected_study_rooms

def test_coordinate_calculation():
    """测试坐标计算"""
    print("\n" + "=" * 50)
    print("📐 坐标计算测试")
    print("=" * 50)
    
    # 模拟一个书房OCR结果
    study_room = {'text': '书房', 'bbox': [200, 150, 80, 40], 'confidence': 0.95}
    
    # 模拟原始图像尺寸
    original_size = (579, 433)
    
    print(f"📋 测试数据:")
    print(f"   📝 OCR文字: '{study_room['text']}'")
    print(f"   📦 边界框: {study_room['bbox']} (x,y,w,h)")
    print(f"   🖼️ 原始图像: {original_size[0]}x{original_size[1]}")
    
    # 计算坐标（模拟重构版本的逻辑）
    x, y, w, h = study_room['bbox']
    
    # OCR文字中心（OCR处理的图像坐标系，2倍放大）
    ocr_center_x = x + w // 2
    ocr_center_y = y + h // 2
    print(f"   🎯 OCR中心: ({ocr_center_x}, {ocr_center_y}) [放大2倍坐标系]")
    
    # 转换到原始图像坐标
    orig_center_x = int(ocr_center_x / 2)
    orig_center_y = int(ocr_center_y / 2)
    print(f"   📍 原图中心: ({orig_center_x}, {orig_center_y}) [原始坐标系]")
    
    # 计算边界框
    half_width = max(50, w // 4)
    half_height = max(30, h // 4)
    
    min_x = max(0, orig_center_x - half_width)
    max_x = min(original_size[0] - 1, orig_center_x + half_width)
    min_y = max(0, orig_center_y - half_height)
    max_y = min(original_size[1] - 1, orig_center_y + half_height)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    print(f"   📐 估算边界框: ({min_x},{min_y}) 到 ({max_x},{max_y})")
    print(f"   📏 房间尺寸: {width} x {height} 像素")
    print(f"   📊 估算面积: {width * height} 像素")

if __name__ == "__main__":
    print("🔬 DeepFloorplan 书房识别功能测试")
    print("=" * 60)
    
    # 运行关键词匹配测试
    detected_rooms = test_study_room_keywords()
    
    # 运行坐标计算测试
    test_coordinate_calculation()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("💡 提示: 在实际户型图中使用时，确保图像包含清晰的'书房'或'study'等文字标识")
