#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
九宫格风水方位可视化工具
在原图上叠加九宫格，显示各方位及对应的八宅八星
直接调用fengshui模块的分析逻辑确保一致性
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
import math
import sys
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from editor.json_io import load_floorplan_json
from fengshui.bazhai_eightstars import analyze_eightstars, HOUSE_DIRECTION_STARS, STAR_INFO
from fengshui.luoshu_missing_corner import analyze_missing_corners_by_room_coverage

def load_json_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_polygon_from_rooms(rooms: List[Dict[str, Any]]) -> List[tuple]:
    """从房间数据创建更精确的外轮廓多边形"""
    if not rooms:
        return []
    
    # 收集所有房间的边界框
    boxes = []
    for room in rooms:
        bbox = room.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1") 
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            boxes.append((x1, y1, x2, y2))
    
    if not boxes:
        return []
    
    # 使用简单的外接矩形方法（与风水分析保持一致）
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    
    # 返回矩形的四个角点，用于可视化边界
    return [(float(min_x), float(min_y)), (float(max_x), float(min_y)), 
            (float(max_x), float(max_y)), (float(min_x), float(max_y))]


def create_detailed_polygon_from_rooms(rooms: List[Dict[str, Any]]) -> List[tuple]:
    """从房间数据创建详细的凸包多边形（仅用于可视化）"""
    if not rooms:
        return []
    
    # 收集所有房间的边界框
    boxes = []
    for room in rooms:
        bbox = room.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1") 
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            boxes.append((x1, y1, x2, y2))
    
    if not boxes:
        return []
    
    # 创建更精确的轮廓点集
    all_points = set()
    
    # 为每个房间添加四个角点
    for x1, y1, x2, y2 in boxes:
        all_points.add((x1, y1))
        all_points.add((x2, y1))
        all_points.add((x2, y2))
        all_points.add((x1, y2))
    
    # 转换为numpy数组进行凸包计算
    points = np.array(list(all_points))
    
    # 使用凸包算法找到外轮廓
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return [(float(x), float(y)) for x, y in hull_points]
    except ImportError:
        # 如果没有scipy，使用简化的方法
        # 按角度排序点来形成近似凸包
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        def angle_from_center(point):
            return np.arctan2(point[1] - center_y, point[0] - center_x)
        
        sorted_points = sorted(points, key=angle_from_center)
        return [(float(x), float(y)) for x, y in sorted_points]

def create_detailed_polygon_from_rooms(rooms: List[Dict[str, Any]]) -> List[tuple]:
    """从房间数据创建详细的凸包多边形（仅用于可视化）"""
    if not rooms:
        return []
    
    # 收集所有房间的边界框
    boxes = []
    for room in rooms:
        bbox = room.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1") 
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            boxes.append((x1, y1, x2, y2))
    
    if not boxes:
        return []
    
    # 创建更精确的轮廓点集
    all_points = set()
    
    # 为每个房间添加四个角点
    for x1, y1, x2, y2 in boxes:
        all_points.add((x1, y1))
        all_points.add((x2, y1))
        all_points.add((x2, y2))
        all_points.add((x1, y2))
    
    # 转换为numpy数组进行凸包计算
    points = np.array(list(all_points))
    
    # 使用凸包算法找到外轮廓
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return [(float(x), float(y)) for x, y in hull_points]
    except ImportError:
        # 如果没有scipy，使用简化的方法
        # 按角度排序点来形成近似凸包
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        def angle_from_center(point):
            return np.arctan2(point[1] - center_y, point[0] - center_x)
        
        sorted_points = sorted(points, key=angle_from_center)
        return [(float(x), float(y)) for x, y in sorted_points]


def get_polygon_bounds(polygon: List[tuple]) -> tuple:
    """获取多边形的边界框"""
    if not polygon:
        return 0, 0, 100, 100
    
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def get_direction_stars_mapping(doc, gua: str = None) -> Dict[str, str]:
    """获取方位到星位的映射，使用fengshui模块的逻辑"""
    if gua:
        from fengshui.bazhai_eightstars import GUA_DIRECTION_STARS
        return GUA_DIRECTION_STARS.get(gua, {})
    else:
        house_orientation = getattr(doc, 'house_orientation', '坐北朝南')
        return HOUSE_DIRECTION_STARS.get(house_orientation, {})

def get_luoshu_grid_positions() -> Dict[str, Tuple[int, int]]:
    """返回九宫格方位映射 (上北下南左西右东)"""
    # 根据图像坐标系：上方是北(y=0)，下方是南(y=2)，左边是西(x=0)，右边是东(x=2)
    return {
        "西北": (0, 0), "北": (1, 0), "东北": (2, 0),
        "西": (0, 1),   "中": (1, 1), "东": (2, 1),
        "西南": (0, 2), "南": (1, 2), "东南": (2, 2)
    }

def cv2_to_pil(cv2_image):
    """将OpenCV图像转换为PIL图像"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    """将PIL图像转换为OpenCV图像"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_chinese_font(size=20):
    """获取中文字体"""
    # 尝试常见的中文字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
    ]
    
    for font_path in font_paths:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # 如果找不到字体，使用默认字体
    try:
        return ImageFont.load_default()
    except:
        return None

def draw_text_with_background(draw, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=3):
    """在PIL图像上绘制带背景的文字"""
    x, y = position
    
    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 绘制背景矩形
    bg_x1 = x - padding
    bg_y1 = y - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + text_height + padding
    
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
    
    # 绘制文字
    draw.text((x, y), text, font=font, fill=text_color)
    
    return text_width, text_height

def get_star_colors():
    """返回八星对应的颜色（与STAR_INFO保持一致）"""
    return {
        "生气": (0, 255, 0),    # 绿色 - 最吉
        "延年": (0, 200, 0),    # 深绿 - 吉
        "天医": (0, 150, 255),  # 橙色 - 吉 
        "伏位": (255, 255, 0),  # 黄色 - 小吉
        "中宫": (128, 128, 128), # 灰色 - 中性
        "绝命": (0, 0, 255),    # 红色 - 大凶
        "五鬼": (0, 0, 200),    # 深红 - 凶
        "六煞": (0, 100, 255),  # 橙红 - 小凶
        "祸害": (0, 50, 200)    # 深橙红 - 凶
    }

def draw_luoshu_grid_with_missing_corners(image, rooms_data, polygon=None, overlay_alpha=0.7, missing_corners=None):
    """在图像上绘制九宫格，并标注缺角信息"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(light_image)
    
    # 创建透明overlay用于绘制九宫格
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 获取房屋边界
    if polygon:
        min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
    else:
        # 如果没有多边形，使用整个图像
        min_x, min_y, max_x, max_y = 0, 0, w, h
    
    # 九宫格尺寸 - 基于实际房屋边界
    house_w = max_x - min_x
    house_h = max_y - min_y
    grid_w = house_w / 3
    grid_h = house_h / 3
    
    directions = get_luoshu_grid_positions()
    
    # 获取字体
    font_size = min(int(house_w), int(house_h)) // 18
    font = get_chinese_font(max(16, font_size))
    small_font = get_chinese_font(max(12, font_size - 2))
    
    # 创建缺角信息映射
    missing_info = {}
    if missing_corners:
        for corner in missing_corners:
            missing_info[corner['direction']] = corner['coverage']
    
    # 如果有多边形，先绘制房屋轮廓
    if polygon:
        polygon_points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon_points, fill=None, outline=(100, 100, 100, 180), width=2)
    
    # 绘制九宫格框架 - 基于房屋实际边界
    for direction, (col, row) in directions.items():
        x1 = min_x + col * grid_w
        y1 = min_y + row * grid_h
        x2 = x1 + grid_w
        y2 = y1 + grid_h
        
        # 检查是否缺角
        is_missing = any(corner['direction'] == direction for corner in missing_corners) if missing_corners else False
        coverage = missing_info.get(direction, 1.0)
        
        # 根据是否缺角选择颜色
        if is_missing:
            edge_color = (255, 0, 0, 255)  # 红色边框表示缺角
            bg_color = (255, 200, 200, 100)  # 浅红色背景
        else:
            edge_color = (0, 0, 0, 255)  # 黑色边框
            bg_color = (200, 255, 200, 50)  # 浅绿色背景
        
        # 绘制背景色
        draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=edge_color, width=2)
        
        # 计算九宫格区域中心
        center_x = x1 + grid_w / 2
        center_y = y1 + grid_h / 2
        
        # 方位名称
        direction_text = direction
        
        # 绘制方位文字（透明背景）
        if font:
            # 计算文字位置使其居中
            bbox = draw.textbbox((0, 0), direction_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = center_x - text_w / 2
            text_y = center_y - text_h / 2 - 10
            
            # 绘制文字（无背景框，使用阴影效果增强可读性）
            # 先绘制白色阴影
            draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
            # 再绘制黑色主文字
            draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
        
        # 如果缺角，显示覆盖率信息（透明背景）
        if is_missing and small_font:
            coverage_text = f"覆盖率: {coverage:.2f}"
            bbox = draw.textbbox((0, 0), coverage_text, font=small_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = center_x - text_w / 2
            text_y = center_y + 10
            
            # 绘制文字（无背景框，使用阴影效果）
            # 白色阴影
            draw.text((text_x + 1, text_y + 1), coverage_text, font=small_font, fill=(255, 255, 255, 180))
            # 红色主文字
            draw.text((text_x, text_y), coverage_text, font=small_font, fill=(255, 0, 0, 255))
    
    # 将透明overlay合成到原图上
    pil_image = pil_image.convert('RGBA')
    result = Image.alpha_composite(pil_image, overlay)
    
    # 转换回OpenCV格式
    return pil_to_cv2(result.convert('RGB'))


def draw_luoshu_grid_only(image, polygon=None, overlay_alpha=0.7):
    """在图像上绘制九宫格（仅显示方位，不显示八星），基于实际房屋轮廓"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(light_image)
    
    # 创建透明overlay用于绘制九宫格
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 获取房屋边界
    if polygon:
        min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
    else:
        # 如果没有多边形，使用整个图像
        min_x, min_y, max_x, max_y = 0, 0, w, h
    
    # 九宫格尺寸 - 基于实际房屋边界
    house_w = max_x - min_x
    house_h = max_y - min_y
    grid_w = house_w / 3
    grid_h = house_h / 3
    
    directions = get_luoshu_grid_positions()
    
    # 获取字体
    font_size = min(int(house_w), int(house_h)) // 18
    font = get_chinese_font(max(16, font_size))
    
    # 如果有多边形，先绘制房屋轮廓
    if polygon:
        polygon_points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon_points, fill=None, outline=(100, 100, 100, 180), width=2)
    
    # 绘制九宫格框架 - 基于房屋实际边界
    for direction, (col, row) in directions.items():
        x1 = min_x + col * grid_w
        y1 = min_y + row * grid_h
        x2 = x1 + grid_w
        y2 = y1 + grid_h
        
        # 绘制更细的边框线
        draw.rectangle([x1, y1, x2, y2], fill=None, outline=(0, 0, 0, 255), width=1)
        
        # 计算九宫格区域中心
        center_x = x1 + grid_w / 2
        center_y = y1 + grid_h / 2
        
        # 方位名称
        direction_text = direction
        
        # 绘制方位文字（透明背景）
        if font:
            # 计算文字位置使其居中
            bbox = draw.textbbox((0, 0), direction_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = center_x - text_w / 2
            text_y = center_y - text_h / 2
            
            # 绘制文字（无背景框，使用阴影效果增强可读性）
            # 先绘制白色阴影
            draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
            # 再绘制黑色主文字
            draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
    
    # 将透明overlay合成到原图上
    pil_image = pil_image.convert('RGBA')
    result = Image.alpha_composite(pil_image, overlay)
    
    # 转换回OpenCV格式
    return pil_to_cv2(result.convert('RGB'))

def draw_bazhai_circle(image, direction_stars_mapping, polygon=None, overlay_alpha=0.7):
    """在图像上绘制八宅八星圆形图，基于实际房屋轮廓"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(light_image)
    
    # 创建透明overlay用于绘制八宅八星
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 获取房屋边界和中心
    if polygon:
        min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        # 半径基于房屋实际尺寸
        radius = min(max_x - min_x, max_y - min_y) / 3
        
        # 绘制房屋轮廓
        polygon_points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon_points, fill=None, outline=(100, 100, 100, 180), width=2)
    else:
        center_x = w / 2
        center_y = h / 2
        radius = min(w, h) / 3
    
    colors = get_star_colors()
    
    # 获取字体
    font_size = int(radius) // 8
    font = get_chinese_font(max(14, font_size))
    small_font = get_chinese_font(max(12, font_size - 2))
    
    # 八个方位的角度（基于图像坐标系：上北下南左西右东）
    direction_angles = {
        "东": 0,      # 右方
        "东南": 45,   # 右下
        "南": 90,     # 下方
        "西南": 135,  # 左下
        "西": 180,    # 左方
        "西北": 225,  # 左上
        "北": 270,    # 上方
        "东北": 315   # 右上
    }
    
    # 绘制八个扇形区域
    for direction, angle in direction_angles.items():
        if direction == "中":  # 跳过中心
            continue
            
        # 计算扇形的起始和结束角度
        start_angle = angle - 22.5
        end_angle = angle + 22.5
        
        # 获取对应的星位和颜色
        star = direction_stars_mapping.get(direction, "未知")
        color = colors.get(star, (128, 128, 128))
        
        # 使用半透明的颜色填充扇形
        alpha_color = color + (150,)  # 添加透明度
        
        # 绘制扇形
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.pieslice(bbox, start_angle, end_angle, fill=alpha_color, outline=(0, 0, 0, 255), width=2)
        
        # 计算文字位置（在扇形中间）
        text_radius = radius * 0.7
        text_angle_rad = math.radians(angle)
        text_x = center_x + text_radius * math.cos(text_angle_rad)
        text_y = center_y - text_radius * math.sin(text_angle_rad)
        
        # 绘制方位和星位文字
        direction_text = direction
        star_text = f"{star}星" if star != "未知" else star
        
        if font:
            # 方位文字 - 白色背景标签
            bbox = draw.textbbox((0, 0), direction_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            label_x = text_x - text_w//2
            label_y = text_y - text_h - 5
            
            # 绘制白色背景标签
            padding = 3
            draw.rectangle([label_x - padding, label_y - padding, 
                          label_x + text_w + padding, label_y + text_h + padding], 
                          fill=(255, 255, 255, 220), outline=(0, 0, 0, 255), width=1)
            draw.text((label_x, label_y), direction_text, font=font, fill=(0, 0, 0, 255))
        
        if small_font:
            # 星位文字 - 白色背景标签
            bbox = draw.textbbox((0, 0), star_text, font=small_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            label_x = text_x - text_w//2
            label_y = text_y + 5
            
            # 绘制白色背景标签
            padding = 2
            draw.rectangle([label_x - padding, label_y - padding, 
                          label_x + text_w + padding, label_y + text_h + padding], 
                          fill=(255, 255, 255, 220), outline=(0, 0, 0, 255), width=1)
            draw.text((label_x, label_y), star_text, font=small_font, fill=(0, 0, 0, 255))
    
    # 绘制中心圆
    center_radius = radius / 4
    star = direction_stars_mapping.get("中", "中宫")
    color = colors.get(star, (128, 128, 128))
    alpha_color = color + (150,)
    
    center_bbox = [center_x - center_radius, center_y - center_radius, 
                   center_x + center_radius, center_y + center_radius]
    draw.ellipse(center_bbox, fill=alpha_color, outline=(0, 0, 0, 255), width=2)
    
    # 中心文字
    if font:
        center_text = "中宫"
        bbox = draw.textbbox((0, 0), center_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        label_x = center_x - text_w//2
        label_y = center_y - text_h//2
        
        # 绘制白色背景标签
        padding = 3
        draw.rectangle([label_x - padding, label_y - padding, 
                      label_x + text_w + padding, label_y + text_h + padding], 
                      fill=(255, 255, 255, 220), outline=(0, 0, 0, 255), width=1)
        draw.text((label_x, label_y), center_text, font=font, fill=(0, 0, 0, 255))
    
    # 将透明overlay合成到原图上
    pil_image = pil_image.convert('RGBA')
    result = Image.alpha_composite(pil_image, overlay)
    
    # 转换回OpenCV格式
    return pil_to_cv2(result.convert('RGB'))

def draw_room_positions(image, rooms_data):
    """在图像上标注房间位置，样式淡化以不抢夺主要内容"""
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 获取字体 - 使用较小的字体
    font = get_chinese_font(12)
    
    for room in rooms_data:
        if 'center' in room:
            center_x = int(room['center']['x'])
            center_y = int(room['center']['y'])
            room_text = f"{room.get('type', 'Unknown')}{room.get('index', '')}"
            
            # 使用PIL绘制房间标签 - 无背景框，使用阴影效果
            if font:
                # 白色阴影
                draw.text((center_x - 24, center_y - 29), room_text, font=font, fill=(255, 255, 255, 150))
                # 深灰色主文字
                draw.text((center_x - 25, center_y - 30), room_text, font=font, fill=(64, 64, 64, 255))
    
    # 转换回OpenCV格式并绘制圆点
    result = pil_to_cv2(pil_image)
    
    # 绘制房间中心点 - 使用更淡的颜色
    for room in rooms_data:
        if 'center' in room:
            center_x = int(room['center']['x'])
            center_y = int(room['center']['y'])
            
            # 绘制房间中心点 - 淡化颜色
            cv2.circle(result, (center_x, center_y), 4, (200, 150, 200), -1)
            cv2.circle(result, (center_x, center_y), 6, (150, 150, 150), 1)
    
    return result

def add_legend(image):
    """添加图例说明"""
    h, w = image.shape[:2]
    legend_height = 150
    legend_width = w
    
    # 创建图例区域
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    legend[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文
    pil_legend = cv2_to_pil(legend)
    draw = ImageDraw.Draw(pil_legend)
    
    # 获取字体
    title_font = get_chinese_font(24)
    text_font = get_chinese_font(16)
    
    # 标题
    title = "八宅八星图例 (坐北朝南)"
    if title_font:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = bbox[2] - bbox[0]
        draw.text((w//2 - title_w//2, 10), title, font=title_font, fill=(255, 255, 255))
    
    # 星位说明
    stars_info = [
        ("生气星", "最吉", (0, 255, 0)),
        ("延年星", "吉", (0, 200, 0)),
        ("天医星", "吉", (0, 150, 255)),
        ("伏位星", "小吉", (255, 255, 0)),
        ("绝命星", "大凶", (0, 0, 255)),
        ("五鬼星", "凶", (0, 0, 200)),
        ("六煞星", "小凶", (0, 100, 255)),
        ("祸害星", "凶", (0, 50, 200))
    ]
    
    # 转换回OpenCV绘制颜色块
    legend_cv2 = pil_to_cv2(pil_legend)
    
    start_x = 50
    start_y = 60
    col_width = w // 4
    
    for i, (star, desc, color) in enumerate(stars_info):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 绘制颜色块
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), color, -1)
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), (255, 255, 255), 1)
    
    # 再次转换为PIL绘制文字
    pil_legend = cv2_to_pil(legend_cv2)
    draw = ImageDraw.Draw(pil_legend)
    
    for i, (star, desc, color) in enumerate(stars_info):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 文字
        text = f"{star} ({desc})"
        if text_font:
            draw.text((x + 30, y + 2), text, font=text_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    final_legend = pil_to_cv2(pil_legend)
    
    # 合并图例和原图
    result = np.vstack([image, final_legend])
    return result

def create_combined_visualization(image, rooms_data, direction_stars_mapping, polygon=None, missing_corners=None):
    """创建组合可视化图像：九宫格图 + 八宅八星圆形图，包含缺角信息"""
    h, w = image.shape[:2]
    
    # 创建两个分离的图像
    if missing_corners:
        luoshu_image = draw_luoshu_grid_with_missing_corners(image.copy(), rooms_data, polygon, missing_corners=missing_corners)
    else:
        luoshu_image = draw_luoshu_grid_only(image.copy(), polygon)
    bazhai_image = draw_bazhai_circle(image.copy(), direction_stars_mapping, polygon)
    
    # 标注房间位置到两个图像上
    luoshu_image = draw_room_positions(luoshu_image, rooms_data)
    bazhai_image = draw_room_positions(bazhai_image, rooms_data)
    
    # 水平拼接两张图
    combined_image = np.hstack([luoshu_image, bazhai_image])
    
    # 添加标题
    combined_h, combined_w = combined_image.shape[:2]
    title_height = 80
    title_area = np.zeros((title_height, combined_w, 3), dtype=np.uint8)
    title_area[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文标题
    pil_title = cv2_to_pil(title_area)
    draw = ImageDraw.Draw(pil_title)
    
    title_font = get_chinese_font(36)
    if title_font:
        # 左侧标题
        if missing_corners:
            left_title = "九宫格缺角分析"
        else:
            left_title = "九宫格方位图"
        bbox = draw.textbbox((0, 0), left_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        left_center_x = combined_w // 4
        draw.text((left_center_x - title_w//2, 20), left_title, font=title_font, fill=(255, 255, 255))
        
        # 右侧标题
        right_title = "八宅八星图"
        bbox = draw.textbbox((0, 0), right_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        right_center_x = combined_w * 3 // 4
        draw.text((right_center_x - title_w//2, 20), right_title, font=title_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    title_cv2 = pil_to_cv2(pil_title)
    
    # 合并标题和图像
    final_image = np.vstack([title_cv2, combined_image])
    
    return final_image


def create_combined_visualization_old(image, rooms_data, direction_stars_mapping, polygon=None):
    """创建组合可视化图像：九宫格图 + 八宅八星圆形图"""
    h, w = image.shape[:2]
    
    # 创建两个分离的图像
    luoshu_image = draw_luoshu_grid_only(image.copy(), polygon)
    bazhai_image = draw_bazhai_circle(image.copy(), direction_stars_mapping, polygon)
    
    # 标注房间位置到两个图像上
    luoshu_image = draw_room_positions(luoshu_image, rooms_data)
    bazhai_image = draw_room_positions(bazhai_image, rooms_data)
    
    # 水平拼接两张图
    combined_image = np.hstack([luoshu_image, bazhai_image])
    
    # 添加标题
    combined_h, combined_w = combined_image.shape[:2]
    title_height = 80
    title_area = np.zeros((title_height, combined_w, 3), dtype=np.uint8)
    title_area[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文标题
    pil_title = cv2_to_pil(title_area)
    draw = ImageDraw.Draw(pil_title)
    
    title_font = get_chinese_font(36)
    if title_font:
        # 左侧标题
        left_title = "九宫格方位图"
        bbox = draw.textbbox((0, 0), left_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        left_center_x = combined_w // 4
        draw.text((left_center_x - title_w//2, 20), left_title, font=title_font, fill=(255, 255, 255))
        
        # 右侧标题
        right_title = "八宅八星图"
        bbox = draw.textbbox((0, 0), right_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        right_center_x = combined_w * 3 // 4
        draw.text((right_center_x - title_w//2, 20), right_title, font=title_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    title_cv2 = pil_to_cv2(pil_title)
    
    # 合并标题和图像
    final_image = np.vstack([title_cv2, combined_image])
    
    return final_image

def add_legend(image, direction_stars_mapping, missing_corners=None):
    """添加图例说明，包含星位映射和缺角信息"""
    h, w = image.shape[:2]
    base_legend_height = 150
    extra_height = 80 if missing_corners else 0
    legend_height = base_legend_height + extra_height
    legend_width = w
    
    # 创建图例区域
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    legend[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文
    pil_legend = cv2_to_pil(legend)
    draw = ImageDraw.Draw(pil_legend)
    
    # 获取字体
    title_font = get_chinese_font(24)
    text_font = get_chinese_font(16)
    small_font = get_chinese_font(14)
    
    # 标题
    title = "八宅八星图例"
    if title_font:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = bbox[2] - bbox[0]
        draw.text((w//2 - title_w//2, 10), title, font=title_font, fill=(255, 255, 255))
    
    # 根据实际的星位映射创建图例
    star_counts = {}
    for direction, star in direction_stars_mapping.items():
        if star and star != "未知":
            star_counts[star] = star_counts.get(star, 0) + 1
    
    # 获取星位信息
    colors = get_star_colors()
    
    # 转换回OpenCV绘制颜色块
    legend_cv2 = pil_to_cv2(pil_legend)
    
    start_x = 50
    start_y = 60
    col_width = w // 4
    
    displayed_stars = []
    for star in star_counts.keys():
        if star in STAR_INFO:
            nature, suggestion = STAR_INFO[star]
            color = colors.get(star, (128, 128, 128))
            displayed_stars.append((star, nature, color))
    
    for i, (star, nature, color) in enumerate(displayed_stars):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 绘制颜色块
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), color, -1)
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), (255, 255, 255), 1)
    
    # 再次转换为PIL绘制文字
    pil_legend = cv2_to_pil(legend_cv2)
    draw = ImageDraw.Draw(pil_legend)
    
    for i, (star, nature, color) in enumerate(displayed_stars):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 文字
        text = f"{star}星 ({nature})"
        if text_font:
            draw.text((x + 30, y + 2), text, font=text_font, fill=(255, 255, 255))
    
    # 如果有缺角信息，添加缺角说明
    if missing_corners and small_font:
        missing_y = start_y + 80
        if title_font:
            missing_title = "缺角分析 (覆盖率 < 60%)"
            bbox = draw.textbbox((0, 0), missing_title, font=title_font)
            title_w = bbox[2] - bbox[0]
            draw.text((w//2 - title_w//2, missing_y), missing_title, font=title_font, fill=(255, 255, 255))
        
        # 绘制缺角信息
        missing_start_y = missing_y + 35
        for i, corner in enumerate(missing_corners):
            x = start_x + (i % 6) * (col_width * 0.8)
            y = missing_start_y + (i // 6) * 25
            
            text = f"{corner['direction']}方: {corner['coverage']:.2f}"
            # 绘制红色标记
            cv2.rectangle(legend_cv2, (int(x), int(y)), (int(x) + 15, int(y) + 15), (0, 0, 255), -1)
            cv2.rectangle(legend_cv2, (int(x), int(y)), (int(x) + 15, int(y) + 15), (255, 255, 255), 1)
            
            # 重新转换为PIL绘制文字
            pil_legend = cv2_to_pil(legend_cv2)
            draw = ImageDraw.Draw(pil_legend)
            draw.text((x + 20, y), text, font=small_font, fill=(255, 255, 255))
            legend_cv2 = pil_to_cv2(pil_legend)
    
    # 转换回OpenCV格式
    final_legend = pil_to_cv2(pil_legend)
    
    # 合并图例和原图
    result = np.vstack([image, final_legend])
    return result


def add_legend_old(image, direction_stars_mapping):
    """添加图例说明，使用实际的星位映射"""
    h, w = image.shape[:2]
    legend_height = 150
    legend_width = w
    
    # 创建图例区域
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    legend[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文
    pil_legend = cv2_to_pil(legend)
    draw = ImageDraw.Draw(pil_legend)
    
    # 获取字体
    title_font = get_chinese_font(24)
    text_font = get_chinese_font(16)
    
    # 标题
    title = "八宅八星图例"
    if title_font:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = bbox[2] - bbox[0]
        draw.text((w//2 - title_w//2, 10), title, font=title_font, fill=(255, 255, 255))
    
    # 根据实际的星位映射创建图例
    star_counts = {}
    for direction, star in direction_stars_mapping.items():
        if star and star != "未知":
            star_counts[star] = star_counts.get(star, 0) + 1
    
    # 获取星位信息
    colors = get_star_colors()
    
    # 转换回OpenCV绘制颜色块
    legend_cv2 = pil_to_cv2(pil_legend)
    
    start_x = 50
    start_y = 60
    col_width = w // 4
    
    displayed_stars = []
    for star in star_counts.keys():
        if star in STAR_INFO:
            nature, suggestion = STAR_INFO[star]
            color = colors.get(star, (128, 128, 128))
            displayed_stars.append((star, nature, color))
    
    for i, (star, nature, color) in enumerate(displayed_stars):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 绘制颜色块
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), color, -1)
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), (255, 255, 255), 1)
    
    # 再次转换为PIL绘制文字
    pil_legend = cv2_to_pil(legend_cv2)
    draw = ImageDraw.Draw(pil_legend)
    
    for i, (star, nature, color) in enumerate(displayed_stars):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 文字
        text = f"{star}星 ({nature})"
        if text_font:
            draw.text((x + 30, y + 2), text, font=text_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    final_legend = pil_to_cv2(pil_legend)
    
    # 合并图例和原图
    result = np.vstack([image, final_legend])
    return result

def visualize_luoshu_grid(json_path, output_path=None, gua=None):
    """生成分离的九宫格和八宅八星可视化图像，使用实际的分析逻辑"""
    
    # 加载户型图数据
    try:
        doc = load_floorplan_json(json_path)
        
        # 同时读取原始JSON数据获取多边形信息
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 尝试获取多边形数据
        polygon = (raw_data.get("polygon") or 
                  raw_data.get("floor_polygon") or 
                  raw_data.get("outline"))
        
        # 如果没有多边形数据，从房间数据创建
        if not polygon:
            rooms = raw_data.get("rooms", [])
            polygon = create_polygon_from_rooms(rooms)
            detailed_polygon = create_detailed_polygon_from_rooms(rooms)
        else:
            detailed_polygon = polygon
            
    except Exception as e:
        print(f"加载户型图数据失败: {e}")
        raise
    
    # 获取图像路径
    meta = raw_data.get('meta', {})
    original_image_path = meta.get('original_image')
    result_image_path = meta.get('result_image') or meta.get('output_image')
    
    # 确定使用的图像路径 - 优先使用原始图像作为清晰底图
    image_path = None
    json_dir = Path(json_path).parent
    project_root = json_dir.parent  # 假设JSON在output目录中
    
    # 优先使用原始图像（清晰的户型图）
    if original_image_path:
        # 处理相对路径
        if '\\' in original_image_path or '/' in original_image_path:
            # 相对于项目根目录的路径
            image_path = project_root / original_image_path.replace('\\', '/')
        else:
            image_path = json_dir / original_image_path
    elif result_image_path:
        # 备选：使用结果图
        if '\\' in result_image_path or '/' in result_image_path:
            image_path = project_root / result_image_path.replace('\\', '/')
        else:
            image_path = json_dir / result_image_path
    
    if not image_path or not image_path.exists():
        # 尝试根据JSON文件名推测图像路径
        json_stem = Path(json_path).stem
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = json_dir / f"{json_stem.replace('_result', '')}{ext}"
            if candidate.exists():
                image_path = candidate
                break
    
    if not image_path or not image_path.exists():
        raise FileNotFoundError(f"找不到对应的图像文件，JSON路径: {json_path}")
    
    # 判断使用的图像类型
    if original_image_path and str(image_path).endswith(original_image_path.split('/')[-1]):
        print(f"使用原始户型图: {image_path}")
    else:
        print(f"使用识别结果图: {image_path}")
    
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像文件: {image_path}")
    
    # 准备房间数据进行八宅分析
    rooms = raw_data.get("rooms", [])
    room_data = []
    
    for room in rooms:
        room_type = room.get("type", "未知")  # 基础类型，如"卧室"
        room_index = room.get("index", 1)    # 序号，如1、2、3
        bbox = room.get("bbox", {})
        
        # 生成显示名称：基础类型 + 序号
        display_name = f"{room_type}{room_index}"
        
        if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            room_data.append({
                "bbox": (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]),
                "name": display_name,  # 显示名称，如"卧室1"
                "type": room_type,     # 基础类型，如"卧室"
                "center": ((bbox["x1"] + bbox["x2"]) / 2, 
                          (bbox["y1"] + bbox["y2"]) / 2),
                "area_pixels": room.get('area_pixels', 0),
                "index": room.get('index', 0)
            })
    
    # 执行八宅八星分析获取实际的方位星位映射
    star_analysis = analyze_eightstars(detailed_polygon, room_data, doc, gua)
    
    # 获取方位到星位的映射
    direction_stars_mapping = get_direction_stars_mapping(doc, gua)
    
    print(f"使用八宅八星分析结果:")
    if gua:
        print(f"  命卦: {gua}")
    else:
        house_orientation = getattr(doc, 'house_orientation', '坐北朝南')
        print(f"  房屋朝向: {house_orientation}")
    
    print(f"  方位星位映射:")
    for direction, star in direction_stars_mapping.items():
        print(f"    {direction}: {star}")
    
    # 执行缺角分析
    rooms_for_analysis = []
    for room in rooms:
        bbox = room.get("bbox", {})
        if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            rooms_for_analysis.append({"bbox": bbox})
    
    missing_corners = analyze_missing_corners_by_room_coverage(
        rooms_for_analysis, doc.img_w, doc.img_h, threshold=0.6
    )
    
    if missing_corners:
        print(f"\n缺角分析结果:")
        for corner in missing_corners:
            print(f"    {corner['direction']}方: 覆盖率 {corner['coverage']:.3f}")
    else:
        print(f"\n缺角分析结果: 无明显缺角")
    
    # 创建组合可视化图像（包含缺角信息）
    final_image = create_combined_visualization(image, rooms, direction_stars_mapping, detailed_polygon, missing_corners)
    
    # 添加图例（包含缺角信息）
    final_image = add_legend(final_image, direction_stars_mapping, missing_corners)
    
    # 保存结果
    if not output_path:
        output_path = json_dir / f"{Path(json_path).stem}_luoshu_grid.png"
    
    cv2.imwrite(str(output_path), final_image)
    print(f"组合可视化图已保存至: {output_path}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='生成分离的九宫格和八宅八星可视化图')
    parser.add_argument('json_path', help='输入的JSON文件路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    parser.add_argument('--gua', help='命卦（如：坎、震、巽、离、坤、乾、兑、艮）')
    
    args = parser.parse_args()
    
    try:
        output_path = visualize_luoshu_grid(args.json_path, args.output, args.gua)
        print(f"✅ 组合可视化完成: {output_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
