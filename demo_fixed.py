#!/usr/bin/env python3
"""
修改demo.py以直接保存RGB图像，避免matplotlib插值
"""
import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2

# 配置TensorFlow日志级别，完全静音冗长输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误
import warnings
warnings.filterwarnings('ignore')

tf.logging.set_verbosity(tf.logging.ERROR)  # 减少TensorFlow日志

# Configure Chinese font support for matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Disable TF 2.x behavior for compatibility
tf.disable_v2_behavior()

# OCR utilities - use enhanced version with PaddleOCR support
from utils.ocr_enhanced import extract_room_text, fuse_ocr_and_segmentation, set_closet_enabled
from utils.rgb_ind_convertor import floorplan_fuse_map, floorplan_fuse_map_figure

def draw_room_regions_with_info(original_image, enhanced_result, ocr_results, output_path):
    """
    在房间分割结果上绘制房间区域信息（坐标、面积等）- 修复过度分割
    
    Args:
        original_image: 原始图像 (numpy array)
        enhanced_result: 房间分割结果 (numpy array)
        ocr_results: OCR检测结果列表
        output_path: 输出图像路径
    """
    import cv2
    from scipy import ndimage
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # 转换为PIL图像以便处理中文
    pil_image = Image.fromarray(original_image)
    draw = ImageDraw.Draw(pil_image)
    
    # 定义房间标签对应的颜色和名称 (RGB格式，PIL使用RGB)
    room_info = {
        1: {'name': '未分类', 'color': (128, 128, 128), 'emoji': '🏠'},
        2: {'name': '卫生间', 'color': (0, 0, 255), 'emoji': '🚿'},
        3: {'name': '客厅/餐厅', 'color': (0, 255, 0), 'emoji': '🛋️'},
        4: {'name': '卧室', 'color': (255, 255, 0), 'emoji': '🛏️'},
        5: {'name': '玄关/大厅', 'color': (128, 0, 128), 'emoji': '🚪'},
        6: {'name': '阳台', 'color': (255, 0, 255), 'emoji': '🌿'},
        7: {'name': '厨房', 'color': (255, 165, 0), 'emoji': '🍳'},
    }
    
    # 尝试加载中文字体
    try:
        # Windows系统字体路径
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        font = ImageFont.truetype(font_path, 16)
        small_font = ImageFont.truetype(font_path, 14)
    except:
        try:
            # 备选字体
            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
            font = ImageFont.truetype(font_path, 16)
            small_font = ImageFont.truetype(font_path, 14)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
    print(f"\n🎯 绘制房间区域和信息标注:")
    
    # 获取图像尺寸
    h_orig, w_orig = original_image.shape[:2]
    
    # 缩放分割结果到原始图像尺寸
    if enhanced_result.shape != (h_orig, w_orig):
        from scipy.ndimage import zoom
        scale_y = h_orig / enhanced_result.shape[0]
        scale_x = w_orig / enhanced_result.shape[1]
        enhanced_result = zoom(enhanced_result, (scale_y, scale_x), order=0)
    
    # 🔧 合并同类型房间区域的新算法
    def merge_room_regions(mask, room_label):
        """智能合并同类型房间区域，减少过度分割"""
        if not np.any(mask):
            return []
        
        # 根据房间类型选择不同的合并策略
        if room_label == 2:  # 卫生间 - 通常较小，使用保守合并
            structure = np.ones((3, 3))
            iterations = 1
            min_area = 300  # 较小的最小面积要求
        elif room_label in [3, 4]:  # 客厅、卧室 - 通常较大，使用激进合并
            structure = np.ones((7, 7))
            iterations = 3
            min_area = 2000  # 较大的最小面积要求
        elif room_label == 7:  # 厨房 - 中等大小
            structure = np.ones((5, 5))
            iterations = 2
            min_area = 800
        else:  # 其他房间类型
            structure = np.ones((4, 4))
            iterations = 2
            min_area = 500
        
        # 形态学闭运算，连接nearby的同类型区域
        closed_mask = ndimage.binary_closing(mask, structure=structure, iterations=iterations)
        
        # 填充小孔洞
        filled_mask = ndimage.binary_fill_holes(closed_mask)
        
        # 再次进行开运算，平滑边界
        opened_mask = ndimage.binary_opening(filled_mask, structure=np.ones((2, 2)))
        
        # 连通组件分析
        labeled_mask, num_features = ndimage.label(opened_mask)
        
        valid_regions = []
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_mask == region_id)
            
            # 检查区域大小
            area = np.sum(region_mask)
            if area < min_area:
                continue
            
            # 检查与原始mask的重叠度
            overlap = np.sum(mask & region_mask)
            overlap_ratio = overlap / area
            
            # 只保留重叠度足够高的区域
            if overlap_ratio > 0.4:  # 至少40%重叠
                valid_regions.append(region_mask)
        
        return valid_regions
    
    # 统计合并后的房间区域
    room_regions = []
    room_count = {}
    
    # 遍历每种房间类型
    for room_label, info in room_info.items():
        room_name = info['name']
        room_color = info['color']
        room_emoji = info['emoji']
        
        # 找到该类型的所有像素
        mask = (enhanced_result == room_label)
        if not np.any(mask):
            continue
        
        # 使用智能合并算法
        merged_regions = merge_room_regions(mask, room_label)
        
        for region_mask in merged_regions:
            # 计算边界框
            y_coords, x_coords = np.where(region_mask)
            if len(y_coords) == 0:
                continue
                
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            w, h = x2 - x1 + 1, y2 - y1 + 1
            area = np.sum(region_mask)
            
            room_regions.append({
                'room_type': room_name,
                'room_label': room_label,
                'color': room_color,
                'emoji': room_emoji,
                'bbox': (x1, y1, x2, y2),
                'area': area,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            })
            
            # 统计房间数量
            room_count[room_name] = room_count.get(room_name, 0) + 1
    
    print(f"📸 共识别出 {len(room_regions)} 个合理的房间区域")
    
    # 绘制每个区域
    for i, region in enumerate(room_regions):
        room_type = region['room_type']
        emoji = region['emoji']
        color = region['color']
        x1, y1, x2, y2 = region['bbox']
        area = region['area']
        w, h = x2 - x1, y2 - y1
        center_x, center_y = region['center']
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 创建信息文本
        info_lines = [
            f"{emoji} {room_type}",
            f"坐标范围: ({x1}, {y1}) -> ({x2}, {y2})",
            f"尺寸: {w} × {h} 像素",
            f"面积: {area} 像素²",
            f"中心点: ({center_x}, {center_y})"
        ]
        
        # 计算信息框位置 - 智能避让
        info_x = min(x2 + 10, w_orig - 220)
        info_y = y1
        
        line_height = 18
        info_box_width = 200
        info_box_height = len(info_lines) * line_height + 12
        
        # 确保信息框不超出图像边界
        if info_y + info_box_height > h_orig:
            info_y = max(0, h_orig - info_box_height)
        if info_x < 0:
            info_x = max(0, x1 - info_box_width - 10)
        
        # 绘制半透明背景框
        bbox_bg = Image.new('RGBA', (info_box_width, info_box_height), (0, 0, 0, 150))
        pil_image.paste(bbox_bg, (info_x, info_y), bbox_bg)
        
        # 绘制边框
        draw.rectangle([info_x, info_y, info_x + info_box_width, info_y + info_box_height], 
                      outline=color, width=2)
        
        # 绘制文本信息
        for idx, line in enumerate(info_lines):
            text_y = info_y + 8 + idx * line_height
            draw.text((info_x + 8, text_y), line, fill=(255, 255, 255), font=small_font)
        
        # 在房间中心绘制序号
        draw.text((center_x - 8, center_y - 8), str(i+1), fill=(255, 255, 255), font=font)
        
        # 控制台输出
        print(f"   {i+1}. {emoji} {room_type}")
        print(f"      📍 坐标范围: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"      📏 尺寸: {w} × {h} 像素")
        print(f"      📐 面积: {area} 像素²")
        print(f"      🎯 中心点: ({center_x}, {center_y})")
        print()
    
    # 打印房间统计
    print(f"📊 房间类型统计:")
    for room_type, count in room_count.items():
        emoji = next(info['emoji'] for info in room_info.values() if info['name'] == room_type)
        print(f"   {emoji} {room_type}: {count} 个")
    
    # 保存图像
    pil_image.save(output_path)
    print(f"📸 房间区域标注图像已保存: {output_path}")
    h, w = enhanced_result.shape[:2]
    room_regions = []
    
    # 分析每个房间标签
    for label in range(1, 8):  # 1-7 对应不同房间类型
        if label in [9, 10]:  # 跳过墙体和门窗
            continue
            
        # 找到该标签的所有区域
        mask = (enhanced_result == label)
        if not np.any(mask):
            continue
            
        # 使用连通组件分析找到独立的房间区域
        labeled_regions, num_regions = ndimage.label(mask)
        
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            
            # 计算区域属性
            coords = np.where(region_mask)
            if len(coords[0]) == 0:
                continue
                
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            # 计算区域信息
            area_pixels = np.sum(region_mask)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            # 跳过过小的区域
            if area_pixels < 100:  # 至少100像素
                continue
                
            # 计算中心点
            center_y = int(np.mean(coords[0]))
            center_x = int(np.mean(coords[1]))
            
            room_regions.append({
                'label': label,
                'name': room_info[label]['name'],
                'color': room_info[label]['color'],
                'emoji': room_info[label]['emoji'],
                'bbox': (min_x, min_y, max_x, max_y),
                'center': (center_x, center_y),
                'area': area_pixels,
                'width': width,
                'height': height
            })
    
    print(f"📸 共识别出 {len(room_regions)} 个房间区域")
    
    # 在图像上绘制房间信息
    for i, room in enumerate(room_regions):
        label = room['label']
        name = room['name']
        color = room['color']
        emoji = room['emoji']
        min_x, min_y, max_x, max_y = room['bbox']
        center_x, center_y = room['center']
        area = room['area']
        width = room['width']
        height = room['height']
        
        # 使用PIL绘制边界框
        draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=3)
        
        # 准备房间信息文本
        info_lines = [
            f"{name}",
            f"坐标: ({min_x},{min_y})-({max_x},{max_y})",
            f"尺寸: {width}x{height}px",
            f"面积: {area}px²"
        ]
        
        # 计算文本框大小
        line_height = 20
        max_text_width = 0
        for line in info_lines:
            bbox_text = draw.textbbox((0, 0), line, font=small_font)
            text_width = bbox_text[2] - bbox_text[0]
            max_text_width = max(max_text_width, text_width)
        
        info_box_width = max_text_width + 12
        info_box_height = len(info_lines) * line_height + 8
        
        # 确定文本框位置（尽量在房间区域内部）
        info_x = max(min_x + 5, center_x - info_box_width // 2)
        info_y = max(min_y + 5, center_y - info_box_height // 2)
        
        # 确保文本框不超出图像边界
        if info_x + info_box_width > w:
            info_x = w - info_box_width - 5
        if info_y + info_box_height > h:
            info_y = h - info_box_height - 5
        if info_x < 5:
            info_x = 5
        if info_y < 5:
            info_y = 5
        
        # 绘制半透明背景框
        background_color = (0, 0, 0, 200)  # 半透明黑色
        background_img = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        background_draw = ImageDraw.Draw(background_img)
        background_draw.rectangle([info_x, info_y, info_x + info_box_width, info_y + info_box_height], 
                                fill=background_color)
        
        # 将背景合并到主图像
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, background_img)
        pil_image = pil_image.convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制边框
        draw.rectangle([info_x, info_y, info_x + info_box_width, info_y + info_box_height], 
                      outline=color, width=2)
        
        # 绘制文本信息
        for idx, line in enumerate(info_lines):
            text_y = info_y + 8 + idx * line_height
            draw.text((info_x + 6, text_y), line, fill=(255, 255, 255), font=small_font)
        
        # 在房间中心绘制房间编号
        number_text = str(i+1)
        # 绘制数字边框（黑色描边）
        for offset_x in [-1, 0, 1]:
            for offset_y in [-1, 0, 1]:
                if offset_x != 0 or offset_y != 0:
                    draw.text((center_x - 8 + offset_x, center_y - 8 + offset_y), number_text, fill=(0, 0, 0), font=font)
        # 绘制数字本体（白色）
        draw.text((center_x - 8, center_y - 8), number_text, fill=(255, 255, 255), font=font)
        
        # 打印房间信息
        print(f"   {i+1}. {emoji} {name}")
        print(f"      📍 坐标范围: ({min_x}, {min_y}) -> ({max_x}, {max_y})")
        print(f"      📏 尺寸: {width} × {height} 像素")
        print(f"      📐 面积: {area} 像素²")
        print(f"      🎯 中心点: ({center_x}, {center_y})")
        print()
    
    # 保存结果图像
    pil_image.save(output_path)
    print(f"📸 房间区域标注图像已保存: {output_path}")
    
    # 转换回numpy array返回
    return np.array(pil_image)

def draw_room_detection_boxes(original_image, ocr_results, output_path):
    """
    在原始图像上绘制所有房间检测框和坐标信息
    
    Args:
        original_image: 原始图像 (numpy array)
        ocr_results: OCR检测结果列表
        output_path: 输出图像路径
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    
    # 转换为PIL图像以便处理中文
    pil_image = Image.fromarray(original_image)
    draw = ImageDraw.Draw(pil_image)
    
    # 定义不同房间类型的颜色 (RGB格式，PIL使用RGB)
    room_colors = {
        'bedroom': (255, 255, 0),      # 黄色 - 卧室
        'bathroom': (0, 0, 255),       # 蓝色 - 卫生间  
        'living_room': (0, 255, 0),    # 绿色 - 客厅
        'kitchen': (255, 165, 0),      # 橙色 - 厨房
        'balcony': (255, 0, 255),      # 洋红色 - 阳台
        'hall': (128, 0, 128),         # 紫色 - 玄关/大厅
        'generic': (255, 255, 255)     # 白色 - 其他房间
    }
    
    # 尝试加载中文字体
    try:
        # Windows系统字体路径
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        font = ImageFont.truetype(font_path, 16)
        small_font = ImageFont.truetype(font_path, 12)
    except:
        try:
            # 备选字体
            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
            font = ImageFont.truetype(font_path, 16)
            small_font = ImageFont.truetype(font_path, 12)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
def draw_room_detection_boxes(original_image, ocr_results, output_path):
    """
    在原始图像上绘制所有房间检测框和坐标信息
    
    Args:
        original_image: 原始图像 (numpy array)
        ocr_results: OCR检测结果列表
        output_path: 输出图像路径
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    
    # 转换为PIL图像以便处理中文
    pil_image = Image.fromarray(original_image)
    draw = ImageDraw.Draw(pil_image)
    
    # 定义不同房间类型的颜色 (RGB格式，PIL使用RGB)
    room_colors = {
        'bedroom': (255, 255, 0),      # 黄色 - 卧室
        'bathroom': (0, 0, 255),       # 蓝色 - 卫生间  
        'living_room': (0, 255, 0),    # 绿色 - 客厅
        'kitchen': (255, 165, 0),      # 橙色 - 厨房
        'balcony': (255, 0, 255),      # 洋红色 - 阳台
        'hall': (128, 0, 128),         # 紫色 - 玄关/大厅
        'generic': (128, 128, 128)     # 灰色 - 其他房间
    }
    
    # 尝试加载中文字体
    try:
        # Windows系统字体路径
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        font = ImageFont.truetype(font_path, 16)
        small_font = ImageFont.truetype(font_path, 12)
    except:
        try:
            # 备选字体
            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
            font = ImageFont.truetype(font_path, 16)
            small_font = ImageFont.truetype(font_path, 12)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # 房间类型检测的关键词映射
    room_keywords = {
        'bedroom': ['卧室', 'bedroom', '房间', '睡房'],
        'bathroom': ['卫生间', '洗手间', '厕所', '浴室', 'bathroom', 'toilet', 'wc'],
        'living_room': ['客厅', '餐厅', '起居室', 'living', 'dining'],
        'kitchen': ['厨房', 'kitchen', '厨', '烹饪'],
        'balcony': ['阳台', 'balcony', '露台'],
        'hall': ['玄关', '大厅', '过道', '走廊', '入口', 'hall', 'entrance']
    }
    
    def get_room_type(text):
        """根据文字内容判断房间类型"""
        text_lower = text.lower()
        for room_type, keywords in room_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return room_type
        return 'generic'
    
    print(f"\n🎯 绘制房间检测框和坐标信息:")
    print(f"📸 共检测到 {len(ocr_results)} 个文字区域")
    
    # 在图像上绘制每个检测到的房间
    room_count = {}
    for i, item in enumerate(ocr_results):
        text = item.get('text', '')
        confidence = item.get('confidence', 0)
        bbox = item.get('bbox', (0, 0, 0, 0))
        
        # 跳过置信度过低的检测结果
        if confidence < 0.3:
            continue
            
        room_type = get_room_type(text)
        color = room_colors.get(room_type, room_colors['generic'])
        
        # 统计房间数量
        if room_type not in room_count:
            room_count[room_type] = 0
        room_count[room_type] += 1
        
        # 获取边界框坐标
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # 计算面积（像素面积）
        area_pixels = w * h
        
        # 使用PIL绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 准备标签文字
        room_emoji = {
            'bedroom': '🛏️',
            'bathroom': '🚿', 
            'living_room': '🛋️',
            'kitchen': '🍳',
            'balcony': '🌿',
            'hall': '🚪',
            'generic': '🏠'
        }
        
        emoji = room_emoji.get(room_type, '🏠')
        
        # 创建信息文本（不使用emoji，避免显示问题）
        info_lines = [
            f"{text}",
            f"坐标: ({x1},{y1})-({x2},{y2})",
            f"尺寸: {w:.0f}x{h:.0f}px",
            f"面积: {area_pixels:.0f}px²",
            f"置信度: {confidence:.2f}"
        ]
        
        # 计算信息框位置
        info_x = x2 + 5
        info_y = y1
        
        # 计算文本框大小
        max_text_width = 0
        line_height = 18
        for line in info_lines:
            bbox_text = draw.textbbox((0, 0), line, font=small_font)
            text_width = bbox_text[2] - bbox_text[0]
            max_text_width = max(max_text_width, text_width)
        
        info_box_width = max_text_width + 10
        info_box_height = len(info_lines) * line_height + 10
        
        # 检查边界，如果超出图像范围则调整位置
        img_width, img_height = pil_image.size
        if info_x + info_box_width > img_width:
            info_x = x1 - info_box_width - 5
        if info_y + info_box_height > img_height:
            info_y = img_height - info_box_height - 5
        if info_x < 0:
            info_x = 5
        if info_y < 0:
            info_y = 5
        
        # 绘制半透明背景框
        background_color = (0, 0, 0, 180)  # 半透明黑色
        background_img = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        background_draw = ImageDraw.Draw(background_img)
        background_draw.rectangle([info_x, info_y, info_x + info_box_width, info_y + info_box_height], 
                                fill=background_color)
        
        # 将背景合并到主图像
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, background_img)
        pil_image = pil_image.convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制边框
        draw.rectangle([info_x, info_y, info_x + info_box_width, info_y + info_box_height], 
                      outline=color, width=2)
        
        # 绘制文本信息
        for idx, line in enumerate(info_lines):
            text_y = info_y + 8 + idx * line_height
            draw.text((info_x + 5, text_y), line, fill=(255, 255, 255), font=small_font)
        
        # 在框的中心绘制序号
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        draw.text((center_x - 5, center_y - 8), str(i+1), fill=(255, 255, 255), font=font)
        
        # 打印坐标信息
        print(f"   {i+1}. {emoji} {text}")
        print(f"      🎯 房间类型: {room_type}")
        print(f"      📍 坐标: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"      📏 尺寸: {w}×{h} 像素")
        print(f"      🎚️ 置信度: {confidence:.3f}")
        print()
    
    # 打印房间统计
    print(f"📊 房间类型统计:")
    for room_type, count in room_count.items():
        emoji = room_emoji.get(room_type, '🏠')
        print(f"   {emoji} {room_type}: {count} 个")
    
    # 保存图像
    pil_image.save(output_path)
    print(f"📸 房间检测框图像已保存: {output_path}")
    
    # 转换回numpy array返回
    return np.array(pil_image)

def imread(path, mode='RGB'):
    """Read image using PIL"""
    img = Image.open(path)
    if mode == 'RGB':
        img = img.convert('RGB')
    elif mode == 'L':
        img = img.convert('L')
    return np.array(img)

def imresize(img, size):
    """Resize image using PIL"""
    if len(img.shape) == 2:  # Grayscale
        img_pil = Image.fromarray(img, mode='L')
    else:  # RGB
        img_pil = Image.fromarray(img, mode='RGB')
    
    resized = img_pil.resize((size[1], size[0]), Image.LANCZOS)
    return np.array(resized)

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('im_path', type=str, nargs='?', default='./demo/demo.jpg',
                    help='input image path')
parser.add_argument('--disable_closet', action='store_true',
                    help='map closet predictions to background')

def simple_connected_components(mask):
    """Simple connected component labeling without scipy"""
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0
    
    def flood_fill(start_y, start_x, label):
        """Flood fill algorithm for connected components"""
        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if (y < 0 or y >= h or x < 0 or x >= w or 
                labels[y, x] != 0 or not mask[y, x]):
                continue
            
            labels[y, x] = label
            # Add 4-connected neighbors
            stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current_label += 1
                flood_fill(y, x, current_label)
    
    return labels, current_label

def create_precise_room_area(floorplan, center_x, center_y, room_label, img_h, img_w):
    """精准房间区域生成算法：继承demo.py的优秀算法，适配所有房间类型"""
    h, w = floorplan.shape
    room_name = {1: "储物间", 2: "卫生间", 3: "客厅", 4: "卧室", 5: "玄关", 6: "阳台", 7: "厨房"}.get(room_label, f"房间{room_label}")
    
    print(f"      🏠 智能生成{room_name}区域: 中心({center_x}, {center_y})")
    
    # 首先检查中心点是否在有效区域（非墙壁）
    if floorplan[center_y, center_x] in [9, 10]:
        print(f"      ⚠️ 中心点在墙壁上，寻找附近的有效区域")
        # 寻找附近的非墙壁区域
        found_valid = False
        for radius in range(1, 15):  # 扩大搜索范围
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    new_y, new_x = center_y + dy, center_x + dx
                    if (0 <= new_y < h and 0 <= new_x < w and 
                        floorplan[new_y, new_x] not in [9, 10]):
                        center_x, center_y = new_x, new_y
                        found_valid = True
                        break
                if found_valid:
                    break
            if found_valid:
                break
        
        if not found_valid:
            print(f"      ❌ 无法找到有效的{room_name}中心点")
            return np.zeros((h, w), dtype=bool)
    
    print(f"      ✅ 使用中心点: ({center_x}, {center_y})")
    
    # 使用泛洪算法找到包含中心点的连通区域
    def flood_fill_room(start_x, start_y):
        """找到包含起始点的完整房间区域"""
        visited = np.zeros((h, w), dtype=bool)
        room_mask = np.zeros((h, w), dtype=bool)
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            if (x < 0 or x >= w or y < 0 or y >= h or 
                visited[y, x] or floorplan[y, x] in [9, 10]):
                continue
            
            visited[y, x] = True
            room_mask[y, x] = True
            
            # 添加4连通的邻居
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return room_mask
    
    # 获取包含房间中心的完整房间
    room_mask = flood_fill_room(center_x, center_y)
    room_pixels = np.sum(room_mask)
    
    # 根据房间类型设置最小面积阈值
    min_room_sizes = {
        1: 50,    # 储物间：最小
        2: 80,    # 卫生间：较小
        3: 200,   # 客厅：较大
        4: 150,   # 卧室：中等
        5: 100,   # 玄关：中等偏小
        6: 80,    # 阳台：较小
        7: 100    # 厨房：中等偏小
    }
    
    min_size = min_room_sizes.get(room_label, 100)
    
    if room_pixels < min_size:
        print(f"      ❌ 房间太小({room_pixels}像素 < {min_size})，不适合做{room_name}")
        return np.zeros((h, w), dtype=bool)
    
    print(f"      📏 发现房间区域: {room_pixels} 像素")
    
    # 计算房间的边界框
    room_coords = np.where(room_mask)
    min_y, max_y = np.min(room_coords[0]), np.max(room_coords[0])
    min_x, max_x = np.min(room_coords[1]), np.max(room_coords[1])
    room_width = max_x - min_x + 1
    room_height = max_y - min_y + 1
    
    print(f"      📐 房间边界: ({min_x},{min_y}) 到 ({max_x},{max_y}), 尺寸{room_width}x{room_height}")
    
    # 根据房间类型设置占用比例和理想大小
    room_configs = {
        1: {'max_ratio': 0.8, 'area_ratio': 0.02, 'ideal_shape': 'compact'},     # 储物间：紧凑
        2: {'max_ratio': 0.8, 'area_ratio': 0.04, 'ideal_shape': 'compact'},     # 卫生间：紧凑
        3: {'max_ratio': 0.9, 'area_ratio': 0.12, 'ideal_shape': 'rectangular'}, # 客厅：较大矩形
        4: {'max_ratio': 0.85, 'area_ratio': 0.08, 'ideal_shape': 'rectangular'}, # 卧室：中等矩形
        5: {'max_ratio': 0.8, 'area_ratio': 0.06, 'ideal_shape': 'elongated'},   # 玄关：狭长
        6: {'max_ratio': 0.8, 'area_ratio': 0.05, 'ideal_shape': 'compact'},     # 阳台：紧凑
        7: {'max_ratio': 0.8, 'area_ratio': 0.06, 'ideal_shape': 'square'}       # 厨房：方形
    }
    
    config = room_configs.get(room_label, {'max_ratio': 0.8, 'area_ratio': 0.06, 'ideal_shape': 'square'})
    
    # 根据房间大小确定房间尺寸（不能超过房间的设定比例）
    max_room_width = int(room_width * config['max_ratio'])
    max_room_height = int(room_height * config['max_ratio'])
    
    # 计算理想的房间尺寸
    total_area = h * w
    target_area = min(total_area * config['area_ratio'], room_pixels * config['max_ratio'])
    target_size = int(np.sqrt(target_area))
    
    # 限制房间大小
    min_size = 15
    target_size = max(min_size, min(target_size, min(max_room_width, max_room_height)))
    
    print(f"      🎯 目标{room_name}尺寸: {target_size}x{target_size}")
    
    # 在房间内创建以中心点为中心的房间区域
    half_size = target_size // 2
    
    # 确保房间区域在房间边界内
    room_left = max(min_x, center_x - half_size)
    room_right = min(max_x + 1, center_x + half_size)
    room_top = max(min_y, center_y - half_size)
    room_bottom = min(max_y + 1, center_y + half_size)
    
    # 根据房间类型调整形状
    if config['ideal_shape'] == 'elongated':  # 玄关：狭长形
        # 优先扩展长度方向
        if room_width > room_height:
            room_left = max(min_x, center_x - target_size)
            room_right = min(max_x + 1, center_x + target_size)
        else:
            room_top = max(min_y, center_y - target_size)
            room_bottom = min(max_y + 1, center_y + target_size)
    elif config['ideal_shape'] == 'rectangular':  # 客厅、卧室：矩形
        # 适当扩展为矩形
        if room_width > room_height:
            extra = target_size // 3
            room_left = max(min_x, center_x - half_size - extra)
            room_right = min(max_x + 1, center_x + half_size + extra)
        else:
            extra = target_size // 3
            room_top = max(min_y, center_y - half_size - extra)
            room_bottom = min(max_y + 1, center_y + half_size + extra)
    # 其他类型保持默认的正方形或紧凑形状
    
    # 创建房间掩码，只在房间区域内
    final_room_mask = np.zeros((h, w), dtype=bool)
    
    for y in range(room_top, room_bottom):
        for x in range(room_left, room_right):
            if room_mask[y, x]:  # 只在原始房间区域内
                final_room_mask[y, x] = True
    
    actual_width = room_right - room_left
    actual_height = room_bottom - room_top
    actual_pixels = np.sum(final_room_mask)
    
    print(f"      ✅ {room_name}区域生成完成:")
    print(f"         边界: ({room_left},{room_top}) 到 ({room_right},{room_bottom})")
    print(f"         尺寸: {actual_width}x{actual_height}")
    print(f"         有效像素: {actual_pixels}")
    
    return final_room_mask

def enhance_room_detection(floorplan, ocr_results):
    """精准房间识别系统：继承demo.py的优秀算法，精确识别所有房间类型"""
    from utils.ocr_enhanced import TEXT_LABEL_MAP, ROOM_TYPE_NAMES, ROOM_TYPE_EMOJIS
    
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # 统计不同类型的OCR检测结果
    ocr_detections = {}
    room_text_found = False
    
    if ocr_results:
        for ocr_item in ocr_results:
            text = ocr_item['text'].strip()
            # 检查是否匹配任何房间类型
            for room_text, label in TEXT_LABEL_MAP.items():
                if room_text in text.lower() or room_text == text:
                    if label not in ocr_detections:
                        ocr_detections[label] = []
                    ocr_detections[label].append({
                        'text': text,
                        'original_text': ocr_item['text'],
                        'confidence': ocr_item.get('confidence', 1.0),
                        'bbox': ocr_item['bbox']
                    })
                    room_text_found = True
                    emoji = ROOM_TYPE_EMOJIS.get(label, "🏠")
                    room_name = ROOM_TYPE_NAMES.get(label, f"房间{label}")
                    print(f"{emoji} OCR检测到{room_name}文字: '{text}' (置信度: {ocr_item.get('confidence', 1.0):.3f})")
                    break

    print(f"🎯 OCR检测到 {len(ocr_detections)} 种房间类型，共 {sum(len(items) for items in ocr_detections.values())} 个房间标识")
    if room_text_found:
        print("✅ 使用OCR优先的精准房间识别策略")
    else:
        print("📍 OCR未检测到明确的房间文字，使用空间分析方法...")

    # 精准处理每种房间类型
    for room_label, detections in ocr_detections.items():
        room_name = ROOM_TYPE_NAMES.get(room_label, f"房间{room_label}")
        emoji = ROOM_TYPE_EMOJIS.get(room_label, "🏠")
        
        print(f"\n{emoji} 精准处理{room_name}: {len(detections)} 个OCR检测")
        
        for detection in detections:
            # 选择最高置信度的检测结果
            best_detection = max(detections, key=lambda x: x['confidence']) if len(detections) > 1 else detection
            
            x, y, w, h = best_detection['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            confidence = best_detection['confidence']
            
            print(f"   📍 选择最可靠的{room_name}: '{best_detection['text']}' (置信度: {confidence:.3f})")
            print(f"   🎯 {room_name}中心位置: ({center_x}, {center_y})")
            
            # 使用demo.py的精准算法生成房间区域
            room_mask = create_precise_room_area(enhanced, center_x, center_y, room_label, h, w)
            if np.sum(room_mask) > 0:
                enhanced[room_mask] = room_label
                room_pixels = np.sum(room_mask)
                print(f"   ✅ 生成精准{room_name}区域: {room_pixels} 像素")
            else:
                print(f"   ⚠️ 无法为{room_name}生成有效区域")
            
            # 每种房间类型只处理置信度最高的一个检测结果
            break

    # 对于没有OCR检测到的房间，使用空间分析
    return enhanced, ocr_detections

def apply_ocr_room_detections(floorplan, ocr_detections):
    """应用OCR检测到的所有房间类型，使用精准的房间识别算法"""
    from utils.ocr_enhanced import ROOM_TYPE_NAMES, ROOM_TYPE_EMOJIS
    
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # 为每种房间类型应用精准的OCR检测结果
    for room_label, detections in ocr_detections.items():
        room_name = ROOM_TYPE_NAMES.get(room_label, f"房间{room_label}")
        emoji = ROOM_TYPE_EMOJIS.get(room_label, "🏠")
        
        print(f"{emoji} 精准处理{room_name}识别: {len(detections)} 个检测结果")
        
        # 选择置信度最高的检测结果
        best_detection = max(detections, key=lambda x: x['confidence'])
        bbox = best_detection['bbox']
        confidence = best_detection['confidence']
        text = best_detection['text']
        
        # 应用精准的房间区域生成算法
        x, y, bbox_w, bbox_h = bbox
        
        # 确保边界框在图像范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x1 = max(0, min(x + bbox_w, w))
        y1 = max(0, min(y + bbox_h, h))
        
        if x1 > x and y1 > y:
            print(f"   📍 {room_name}文字位置: ({x},{y})-({x1},{y1}), 置信度: {confidence:.3f}")
            
            # 以OCR文字为中心，生成精准的房间区域
            center_x, center_y = (x + x1) // 2, (y + y1) // 2
            
            # 使用精准房间区域生成算法
            room_mask = create_precise_room_area(enhanced, center_x, center_y, room_label, bbox_h, bbox_w)
            
            if np.sum(room_mask) > 0:
                enhanced[room_mask] = room_label
                room_pixels = np.sum(room_mask)
                print(f"   ✅ 基于OCR精准识别{room_name}: '{text}' (面积: {room_pixels}像素)")
            else:
                print(f"   ⚠️ 无法为{room_name}生成有效区域: '{text}'")
    
    return enhanced



def ind2rgb(ind_im, enable_closet=True):
    """Convert indexed image to RGB"""
    # Use the appropriate color map based on closet setting
    if enable_closet:
        color_map = floorplan_fuse_map_figure
    else:
        # Create a modified map without closet
        color_map = floorplan_fuse_map_figure.copy()
        color_map[1] = color_map[3]  # Map closet to living room color
    
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3), dtype=np.uint8)

    for i, rgb in color_map.items():
        rgb_im[(ind_im==i)] = rgb

    return rgb_im

def main(args):
    enable_closet = not args.disable_closet
    set_closet_enabled(enable_closet)

    # Load input
    im = imread(args.im_path, mode='RGB')
    original_im = im.copy()
    
    # Resize image for network inference
    im = imresize(im, (512, 512))
    
    # For OCR, use larger, enhanced image
    from PIL import Image, ImageEnhance
    ocr_img = Image.fromarray(original_im)
    # Enlarge for better OCR
    ocr_img = ocr_img.resize((ocr_img.width * 2, ocr_img.height * 2), Image.LANCZOS)
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(ocr_img)
    ocr_img = enhancer.enhance(2.5)
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(ocr_img)
    ocr_img = enhancer.enhance(2.0)
    ocr_im = np.array(ocr_img)
    
    # Extract textual room labels using OCR with enhanced image
    ocr_results = extract_room_text(ocr_im)
    print(f"🔍 原始OCR检测到 {len(ocr_results) if ocr_results else 0} 个文字结果")
    
    # Debug: 显示所有OCR检测结果
    if ocr_results:
        print("📝 OCR原始检测结果:")
        for i, item in enumerate(ocr_results):
            text = item.get('text', '')
            confidence = item.get('confidence', 0)
            bbox = item.get('bbox', (0,0,0,0))
            print(f"   {i+1}. 文字: '{text}' | 置信度: {confidence:.3f} | 位置: {bbox}")
    
    # Scale OCR bounding boxes to match segmentation size (512x512)
    if ocr_results:
        scale_x = im.shape[1] / ocr_im.shape[1]
        scale_y = im.shape[0] / ocr_im.shape[0]
        for item in ocr_results:
            x, y, w, h = item['bbox']
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            item['bbox'] = (x, y, w, h)
    
    # Convert to float and normalize for network inference
    im = im.astype(np.float32) / 255.

    # 检测GPU可用性并配置TensorFlow
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0 if hasattr(tf.config, 'experimental') else False
    if not gpu_available:
        try:
            # TF 1.x的GPU检测方法
            from tensorflow.python.client import device_lib
            local_devices = device_lib.list_local_devices()
            gpu_available = any(device.device_type == 'GPU' for device in local_devices)
        except:
            gpu_available = False
    
    print(f"💻 设备状态: {'GPU可用' if gpu_available else 'CPU模式'}")
    
    # Create tensorflow session with optimized configuration
    config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
    )
    # Enable GPU memory growth to avoid allocation issues
    if gpu_available:
        config.gpu_options.allow_growth = True
        print("🚀 使用GPU加速")
    else:
        # Disable GPU if not available
        print("🔧 使用CPU计算")
        
    with tf.Session(config=config) as sess:
            
            # Initialize
            sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Restore pretrained model
            saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
            saver.restore(sess, './pretrained/pretrained_r3d')

            # Get default graph
            graph = tf.get_default_graph()

            # Restore inputs & outputs tensor
            x = graph.get_tensor_by_name('inputs:0')
            room_type_logit = graph.get_tensor_by_name('Cast:0')
            room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

            # Infer results
            [room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],
                                                            feed_dict={x:im.reshape(1,512,512,3)})
            room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

            # Merge results
            floorplan = room_type.copy()
            floorplan[room_boundary==1] = 9
            floorplan[room_boundary==2] = 10
            
            # Use OCR labels to refine room categories
            floorplan = fuse_ocr_and_segmentation(floorplan, ocr_results)
            
            # 使用通用房间识别系统（包含厨房识别）
            floorplan, ocr_detections = enhance_room_detection(floorplan, ocr_results)
            
            # 应用所有OCR检测到的房间类型（精确识别）
            if ocr_detections:
                print("🎯 应用OCR精确房间识别结果:")
                floorplan = apply_ocr_room_detections(floorplan, ocr_detections)
                
            # If closet is disabled, map closet areas to bedroom (2)
            if not enable_closet:
                floorplan[floorplan==1] = 2  # Map closet to bedroom
                    
            # Convert to RGB
            floorplan_rgb = ind2rgb(floorplan, True)  # Always use full color map

            # 确保output目录存在
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save raw RGB result directly using PIL
            base_name = os.path.basename(args.im_path).split('.')[0]
            output_name = os.path.join(output_dir, base_name + '_raw_result.png')
            result_img = Image.fromarray(floorplan_rgb, mode='RGB')
            result_img.save(output_name)
            print(f"📸 原始RGB结果已保存: {output_name}")
            
            # 绘制房间检测框和坐标信息 (OCR文字检测)
            detection_output = os.path.join(output_dir, base_name + '_room_detection.png')
            draw_room_detection_boxes(original_im, ocr_results, detection_output)
            
            # 绘制房间区域和详细信息（房间分割结果）
            regions_output = os.path.join(output_dir, base_name + '_room_regions.png')
            draw_room_regions_with_info(original_im, floorplan, ocr_results, regions_output)
            
            # Also create matplotlib version for comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(original_im)
            plt.title('原始图片')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(floorplan_rgb)
            plt.title('户型分析结果 (绿色=厨房)')
            plt.axis('off')
            
            # Save matplotlib result
            matplotlib_output = os.path.join(output_dir, base_name + '_matplotlib_result.png')
            plt.savefig(matplotlib_output, dpi=300, bbox_inches='tight')
            print(f"📸 Matplotlib结果已保存: {matplotlib_output}")
            
            plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
