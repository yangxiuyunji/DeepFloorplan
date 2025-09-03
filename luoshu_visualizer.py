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

# 八卦宫位映射
DIRECTION_TO_BAGUA = {
    "北": "坎",
    "东北": "艮", 
    "东": "震",
    "东南": "巽",
    "南": "离",
    "西南": "坤",
    "西": "兑",
    "西北": "乾",
    "中": "中宫"
}

from editor.json_io import load_floorplan_json
from fengshui.bazhai_eightstars import analyze_eightstars, HOUSE_DIRECTION_STARS, STAR_INFO
from fengshui.luoshu_missing_corner import analyze_missing_corners_by_room_coverage

# 房屋朝向到宅卦的映射
HOUSE_ORIENTATION_TO_GUA = {
    "坐北朝南": "坎",
    "坐南朝北": "离", 
    "坐东朝西": "震",
    "坐西朝东": "兑",
    "坐东北朝西南": "艮",
    "坐西南朝东北": "坤",
    "坐东南朝西北": "巽",
    "坐西北朝东南": "乾"
}

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

def get_wall_boundary_from_image(boundary_image_path: str) -> tuple:
    """从边界图像中提取墙体的最小外接矩形"""
    try:
        # 读取边界图像
        boundary_img = cv2.imread(boundary_image_path, cv2.IMREAD_GRAYSCALE)
        if boundary_img is None:
            return None
        
        # 使用形态学操作来找到墙体的主要结构
        # 先腐蚀去除细小噪点，再膨胀恢复主要结构
        kernel = np.ones((3,3), np.uint8)
        processed = cv2.morphologyEx(boundary_img, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的轮廓（主要建筑轮廓）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取轮廓的最小外接矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 适当收缩边界以获得更紧凑的范围（如果轮廓就是整个图像边界）
        img_h, img_w = boundary_img.shape
        if x == 0 and y == 0 and w == img_w and h == img_h:
            # 如果轮廓就是整个图像，尝试通过像素密度分析找到更紧凑的边界
            margin = 10  # 从边缘向内收缩的像素数
            x, y = margin, margin
            w, h = img_w - 2*margin, img_h - 2*margin
        
        return x, y, x+w, y+h
        
    except Exception as e:
        print(f"从边界图像提取墙体轮廓失败: {e}")
        return None

def get_optimized_house_boundary(rooms_data, image_width, image_height, boundary_image_path=None):
    """获取优化的房屋边界，考虑房间分布和墙体厚度"""
    
    # 首先尝试从边界图像获取
    if boundary_image_path:
        wall_bounds = get_wall_boundary_from_image(boundary_image_path)
        if wall_bounds:
            return wall_bounds
    
    # 如果没有边界图像或提取失败，从房间数据计算
    if not rooms_data:
        return 0, 0, image_width, image_height
    
    # 从房间数据获取边界
    x_coords = []
    y_coords = []
    
    for room in rooms_data:
        bbox = room.get("bbox", {})
        if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            x_coords.extend([bbox["x1"], bbox["x2"]])
            y_coords.extend([bbox["y1"], bbox["y2"]])
    
    if not x_coords or not y_coords:
        return 0, 0, image_width, image_height
    
    # 计算房间的边界
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # 添加墙体厚度（假设墙体厚度约为图像尺寸的2-3%）
    wall_thickness = max(int(image_width * 0.025), int(image_height * 0.025), 10)
    
    # 扩展边界以包含墙体
    min_x = max(0, min_x - wall_thickness)
    min_y = max(0, min_y - wall_thickness)
    max_x = min(image_width, max_x + wall_thickness)
    max_y = min(image_height, max_y + wall_thickness)
    
    return min_x, min_y, max_x, max_y

def get_direction_stars_mapping(doc, gua: str = None) -> Dict[str, str]:
    """获取方位到星位的映射，使用fengshui模块的逻辑"""
    if gua:
        from fengshui.bazhai_eightstars import GUA_DIRECTION_STARS
        return GUA_DIRECTION_STARS.get(gua, {})
    else:
        house_orientation = getattr(doc, 'house_orientation', '坐北朝南')
        return HOUSE_DIRECTION_STARS.get(house_orientation, {})

def get_direction_from_point(cx: float, cy: float, ox: float, oy: float, north_angle: int = 90) -> str:
    """Convert a point to compass direction considering north angle.
    
    Args:
        cx, cy: Point coordinates
        ox, oy: Origin/center coordinates  
        north_angle: North direction angle (0=East, 90=North, 180=West, 270=South)
        
    Returns:
        Direction name in Chinese
    """
    DIRECTION_NAMES = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    
    dx = cx - ox
    dy = cy - oy
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0  # 0=East, 90=North
    angle = (angle - north_angle + 360.0) % 360.0
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]


def get_bazhai_direction_angles(north_angle: int = 90) -> Dict[str, float]:
    """根据north_angle返回八宅八星方位角度映射
    
    Args:
        north_angle: 北方角度 (0=东, 90=北, 180=西, 270=南)
    Returns:
        方位到角度的映射字典
    """
    # 基础角度偏移量 = north_angle - 90 (因为默认上方是北方，即90度)
    angle_offset = north_angle - 90
    
    # 八个方位的基础角度（假设上方是北方）
    base_angles = {
        "东": 0,      # 右方
        "东南": 45,   # 右下
        "南": 90,     # 下方
        "西南": 135,  # 左下
        "西": 180,    # 左方
        "西北": 225,  # 左上
        "北": 270,    # 上方
        "东北": 315   # 右上
    }
    
    # 应用角度偏移
    adjusted_angles = {}
    for direction, base_angle in base_angles.items():
        adjusted_angles[direction] = (base_angle + angle_offset) % 360
    
    return adjusted_angles


def get_luoshu_grid_positions(north_angle: int = 90) -> Dict[str, Tuple[int, int]]:
    """返回九宫格方位映射，根据north_angle动态调整
    
    Args:
        north_angle: 北方角度 (0=东, 90=北, 180=西, 270=南)
    """
    # 根据north_angle确定上方是什么方向
    if north_angle == 90:
        # 标准情况：上方是北方
        return {
            "西北": (0, 0), "北": (1, 0), "东北": (2, 0),
            "西": (0, 1),   "中": (1, 1), "东": (2, 1),
            "西南": (0, 2), "南": (1, 2), "东南": (2, 2)
        }
    elif north_angle == 270:
        # 上方是南方，下方是北方，右方是西方，左方是东方
        return {
            "东南": (0, 0), "南": (1, 0), "西南": (2, 0),  # 上排：左东南，中南，右西南
            "东": (0, 1),   "中": (1, 1), "西": (2, 1),    # 中排：左东，中中，右西
            "东北": (0, 2), "北": (1, 2), "西北": (2, 2)   # 下排：左东北，中北，右西北
        }
    elif north_angle == 0:
        # 上方是东方
        return {
            "东北": (0, 0), "东": (1, 0), "东南": (2, 0),
            "北": (0, 1),   "中": (1, 1), "南": (2, 1),
            "西北": (0, 2), "西": (1, 2), "西南": (2, 2)
        }
    elif north_angle == 180:
        # 上方是西方
        return {
            "西南": (0, 0), "西": (1, 0), "西北": (2, 0),
            "南": (0, 1),   "中": (1, 1), "北": (2, 1),
            "东南": (0, 2), "东": (1, 2), "东北": (2, 2)
        }
    else:
        # 默认回退到标准情况
        return {
            "西北": (0, 0), "北": (1, 0), "东北": (2, 0),
            "西": (0, 1),   "中": (1, 1), "东": (2, 1),
            "西南": (0, 2), "南": (1, 2), "东南": (2, 2)
        }


def get_direction_from_grid_position(gx: int, gy: int, north_angle: int = 90) -> str:
    """根据九宫格位置获取方向，使用与角度方法一致的逻辑
    
    Args:
        gx, gy: 网格位置 (0-2)
        north_angle: 北方角度
        
    Returns:
        方向名称
    """
    # 将网格位置转换为相对于中心的坐标
    # 网格中心是(1, 1)，所以相对坐标是(gx-1, gy-1)
    dx = gx - 1  # -1, 0, 1 对应 左, 中, 右
    dy = gy - 1  # -1, 0, 1 对应 上, 中, 下
    
    # 处理中心位置
    if dx == 0 and dy == 0:
        return "中"
    
    # 计算角度（PIL坐标系：右为0度，下为90度）
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    
    # 根据north_angle调整角度
    angle = (angle - north_angle + 90 + 360.0) % 360.0  # +90度是因为默认north_angle=90对应上方
    
    # 转换为方向索引
    DIRECTION_NAMES = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]

def cv2_to_pil(cv2_image):
    """将OpenCV图像转换为PIL图像"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    """将PIL图像转换为OpenCV图像"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_dashed_rectangle(draw, bbox, color=(0, 0, 0, 255), width=2, dash_length=8, gap_length=4):
    """在PIL图像上绘制虚线矩形"""
    x1, y1, x2, y2 = bbox
    
    # 绘制四条边的虚线
    # 上边
    draw_dashed_line(draw, (x1, y1), (x2, y1), color, width, dash_length, gap_length)
    # 右边
    draw_dashed_line(draw, (x2, y1), (x2, y2), color, width, dash_length, gap_length)
    # 下边
    draw_dashed_line(draw, (x2, y2), (x1, y2), color, width, dash_length, gap_length)
    # 左边
    draw_dashed_line(draw, (x1, y2), (x1, y1), color, width, dash_length, gap_length)

def get_minimum_enclosing_circle(polygon):
    """计算多边形的最小外接圆"""
    if not polygon:
        return None, None, None
    
    # 转换为numpy数组
    points = np.array(polygon, dtype=np.float32)
    
    # 使用OpenCV计算最小外接圆
    (center_x, center_y), radius = cv2.minEnclosingCircle(points)
    
    return center_x, center_y, radius

def get_minimum_enclosing_circle_from_rooms(rooms_data, image_width, image_height):
    """从房间数据计算最小外接圆"""
    if not rooms_data:
        # 如果没有房间数据，返回图像中心的圆
        center_x = image_width / 2
        center_y = image_height / 2
        radius = min(image_width, image_height) / 3
        return center_x, center_y, radius
    
    # 收集所有房间的角点
    all_points = []
    for room in rooms_data:
        bbox = room.get("bbox", {})
        if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            all_points.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    if not all_points:
        # 如果没有有效点，返回图像中心的圆
        center_x = image_width / 2
        center_y = image_height / 2
        radius = min(image_width, image_height) / 3
        return center_x, center_y, radius
    
    # 计算最小外接圆
    points = np.array(all_points, dtype=np.float32)
    (center_x, center_y), radius = cv2.minEnclosingCircle(points)
    
    return center_x, center_y, radius

def draw_dashed_line(draw, start, end, color=(0, 0, 0, 255), width=2, dash_length=8, gap_length=4):
    """在PIL图像上绘制虚线"""
    x1, y1 = start
    x2, y2 = end
    
    # 计算线段长度和方向
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    
    if length == 0:
        return
    
    # 单位方向向量
    unit_x = dx / length
    unit_y = dy / length
    
    # 绘制虚线段
    current_length = 0
    while current_length < length:
        # 虚线段起点
        start_x = x1 + current_length * unit_x
        start_y = y1 + current_length * unit_y
        
        # 虚线段终点
        end_length = min(current_length + dash_length, length)
        end_x = x1 + end_length * unit_x
        end_y = y1 + end_length * unit_y
        
        # 绘制线段
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
        
        # 移动到下一个虚线段
        current_length += dash_length + gap_length

def get_chinese_font(size=20):
    """获取中文字体，优先使用楷体"""
    # 优先尝试楷体字体路径
    font_paths = [
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
        "C:/Windows/Fonts/SIMKAI.TTF",  # 楷体（大写）
        "C:/Windows/Fonts/kaiti.ttf",   # 其他楷体
        "C:/Windows/Fonts/msyh.ttc",   # 备用：微软雅黑
        "C:/Windows/Fonts/simhei.ttf", # 备用：黑体
        "C:/Windows/Fonts/simsun.ttc", # 备用：宋体
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

def get_kaiti_font(size=20):
    """获取楷体字体"""
    # 优先尝试楷体字体路径
    font_paths = [
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
        "C:/Windows/Fonts/SIMKAI.TTF",  # 楷体（大写）
        "C:/Windows/Fonts/kaiti.ttf",   # 其他楷体
        "C:/Windows/Fonts/msyh.ttc",   # 备用：微软雅黑
        "C:/Windows/Fonts/simsun.ttc", # 备用：宋体
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

def draw_luoshu_grid_with_missing_corners(image, rooms_data, polygon=None, overlay_alpha=0.7, missing_corners=None, original_image_path=None, north_angle=90):
    """在图像上绘制九宫格，显示缺角信息，支持动态朝向"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 创建透明覆盖层
    pil_image = cv2_to_pil(light_image)
    overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 获取房屋边界 - 基于房间分布或多边形
    if polygon:
        min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
        print(f"使用房间边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    else:
        # 如果没有多边形，从房间数据创建边界
        if rooms_data:
            temp_polygon = create_polygon_from_rooms(rooms_data)
            if temp_polygon:
                min_x, min_y, max_x, max_y = get_polygon_bounds(temp_polygon)
                print(f"使用房间数据边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
            else:
                # 如果仍然没有边界，使用整个图像
                min_x, min_y, max_x, max_y = 0, 0, w, h
                print(f"使用整个图像边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
        else:
            # 没有房间数据和多边形，使用整个图像
            min_x, min_y, max_x, max_y = 0, 0, w, h
            print(f"使用整个图像边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    
    # 九宫格尺寸 - 基于确定的边界
    house_w = max_x - min_x
    house_h = max_y - min_y
    grid_w = house_w / 3
    grid_h = house_h / 3
    
    directions = get_luoshu_grid_positions(north_angle)
    
    # 获取字体
    font_size = min(int(house_w), int(house_h)) // 18
    font = get_chinese_font(max(16, font_size))
    small_font = get_chinese_font(max(6, font_size - 6))  # 缺角率字体更小
    bagua_font_size = max(20, font_size + 4)  # 宫位字体更大
    
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
        if bg_color[3] > 0:  # 如果有背景色
            draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=None)
        
        # 绘制虚线边框
        draw_dashed_rectangle(draw, [x1, y1, x2, y2], edge_color, width=2)
        
        # 计算九宫格区域中心
        center_x = x1 + grid_w / 2
        center_y = y1 + grid_h / 2
        
        # 方位名称和宫位名称
        direction_text = direction
        bagua_text = DIRECTION_TO_BAGUA.get(direction, "")
        
        # 绘制方位文字（透明背景）
        if font:
            # 先绘制宫位名称（八卦）- 使用楷体和红色
            if bagua_text and bagua_text != "中宫":
                kaiti_font = get_kaiti_font(bagua_font_size)  # 获取更大的楷体字体
                if kaiti_font:
                    bagua_bbox = draw.textbbox((0, 0), bagua_text, font=kaiti_font)
                    bagua_w = bagua_bbox[2] - bagua_bbox[0]
                    bagua_h = bagua_bbox[3] - bagua_bbox[1]
                    
                    bagua_x = center_x - bagua_w / 2
                    bagua_y = center_y - bagua_h / 2 - 25  # 在方位名称上方
                    
                    # 绘制宫位名称（红色楷体）
                    draw.text((bagua_x + 1, bagua_y + 1), bagua_text, font=kaiti_font, fill=(255, 255, 255, 180))
                    draw.text((bagua_x, bagua_y), bagua_text, font=kaiti_font, fill=(255, 0, 0, 255))
            
            # 为 draw_luoshu_grid_with_missing_corners 函数专用的对角方位处理
            if direction == "东北":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：右上角是东北
                    text_x = grid_max_x + margin
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 270:  # 上南下北：左下角是东北
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_max_y + margin
                elif north_angle == 0:  # 右北左南：右上角是东北
                    text_x = grid_max_x + margin
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 180:  # 左北右南：左下角是东北
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_max_y + margin
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "西北":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：左上角是西北
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 270:  # 上南下北：右下角是西北
                    text_x = grid_max_x + margin
                    text_y = grid_max_y + margin
                elif north_angle == 0:  # 右北左南：右下角是西北
                    text_x = grid_max_x + margin
                    text_y = grid_max_y + margin
                elif north_angle == 180:  # 左北右南：左上角是西北
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_min_y - text_h - margin
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "东南":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：右下角是东南
                    text_x = grid_max_x + margin
                    text_y = grid_max_y + margin
                elif north_angle == 270:  # 上南下北：左上角是东南
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 0:  # 右北左南：左上角是东南
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 180:  # 左北右南：右下角是东南
                    text_x = grid_max_x + margin
                    text_y = grid_max_y + margin
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "西南":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：左下角是西南
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_max_y + margin
                elif north_angle == 270:  # 上南下北：右上角是西南
                    text_x = grid_max_x + margin
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 0:  # 右北左南：左下角是西南
                    text_x = grid_min_x - text_w - margin
                    text_y = grid_max_y + margin
                elif north_angle == 180:  # 左北右南：右上角是西南
                    text_x = grid_max_x + margin
                    text_y = grid_min_y - text_h - margin
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "北":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：正上方是北
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 270:  # 上南下北：正下方是北
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_max_y + margin
                elif north_angle == 0:  # 右北左南：正左方是北
                    text_x = grid_min_x - text_w - margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                elif north_angle == 180:  # 左北右南：正右方是北
                    text_x = grid_max_x + margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "南":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：正下方是南
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_max_y + margin
                elif north_angle == 270:  # 上南下北：正上方是南
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 0:  # 右北左南：正右方是南
                    text_x = grid_max_x + margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                elif north_angle == 180:  # 左北右南：正左方是南
                    text_x = grid_min_x - text_w - margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "东":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：正右方是东
                    text_x = grid_max_x + margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                elif north_angle == 270:  # 上南下北：正左方是东
                    text_x = grid_min_x - text_w - margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                elif north_angle == 0:  # 右北左南：正上方是东
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_min_y - text_h - margin
                elif north_angle == 180:  # 左北右南：正下方是东
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_max_y + margin
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            elif direction == "西":
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                margin = 15
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                if north_angle == 90:  # 标准朝向：正左方是西
                    text_x = grid_min_x - text_w - margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                elif north_angle == 270:  # 上南下北：正右方是西
                    text_x = grid_max_x + margin
                    text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                elif north_angle == 0:  # 右北左南：正下方是西
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_max_y + margin
                elif north_angle == 180:  # 左北右南：正上方是西
                    text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                    text_y = grid_min_y - text_h - margin
                else:
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
            else:
                # 其他方向（比如"中"）保持原有逻辑
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = center_x - text_w / 2
                text_y = center_y - text_h / 2
            
            # 绘制方位文字（在grid外或中心）
            if direction != "中":
                # 绘制阴影
                draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
                # 绘制主文字
                draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
            else:
                # 中宫位置仍然绘制在格子中心
                # 绘制阴影
                draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
                # 绘制主文字
                draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
            
            # 如果是缺角，绘制缺角率信息
            if is_missing and small_font:
                # 缺角率 = 1 - 覆盖率
                missing_rate = 1.0 - coverage
                missing_rate_text = f"{missing_rate:.1%}"
                missing_rate_bbox = draw.textbbox((0, 0), missing_rate_text, font=small_font)
                missing_rate_w = missing_rate_bbox[2] - missing_rate_bbox[0]
                missing_rate_h = missing_rate_bbox[3] - missing_rate_bbox[1]
                
                missing_rate_x = center_x - missing_rate_w / 2
                missing_rate_y = center_y + 10  # 在中心稍下方
                
                # 绘制缺角率信息（红色）
                draw.text((missing_rate_x + 1, missing_rate_y + 1), missing_rate_text, font=small_font, fill=(255, 255, 255, 180))
                draw.text((missing_rate_x, missing_rate_y), missing_rate_text, font=small_font, fill=(255, 0, 0, 255))
    
    # 合并图像
    result = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    
    # 转换回OpenCV格式
    result_cv2 = pil_to_cv2(result.convert('RGB'))
    
    return result_cv2
    """在图像上绘制九宫格，并标注缺角信息"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(light_image)
    
    # 创建透明overlay用于绘制九宫格
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 获取房屋边界 - 基于房间分布
    if polygon:
        min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
        print(f"使用房间边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    else:
        # 如果没有多边形，从房间数据创建边界
        if rooms_data:
            temp_polygon = create_polygon_from_rooms(rooms_data)
            if temp_polygon:
                min_x, min_y, max_x, max_y = get_polygon_bounds(temp_polygon)
                print(f"使用房间数据边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
            else:
                min_x, min_y, max_x, max_y = 0, 0, w, h
                print(f"使用整个图像边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
        else:
            min_x, min_y, max_x, max_y = 0, 0, w, h
            print(f"使用整个图像边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    
    # 九宫格尺寸 - 基于确定的边界
    house_w = max_x - min_x
    house_h = max_y - min_y
    grid_w = house_w / 3
    grid_h = house_h / 3
    
    directions = get_luoshu_grid_positions(north_angle)
    
    # 获取字体
    font_size = min(int(house_w), int(house_h)) // 18
    font = get_chinese_font(max(16, font_size))
    small_font = get_chinese_font(max(8, font_size - 4))  # 缺角率字体更小
    bagua_font_size = max(20, font_size + 4)  # 宫位字体更大
    
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
        if bg_color[3] > 0:  # 如果有背景色
            draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=None)
        
        # 绘制虚线边框
        draw_dashed_rectangle(draw, [x1, y1, x2, y2], edge_color, width=2)
        
        # 计算九宫格区域中心
        center_x = x1 + grid_w / 2
        center_y = y1 + grid_h / 2
        
        # 方位名称和宫位名称
        direction_text = direction
        bagua_text = DIRECTION_TO_BAGUA.get(direction, "")
        
        # 绘制方位文字（透明背景）
        if font:
            # 先绘制宫位名称（八卦）- 使用楷体和红色
            if bagua_text and bagua_text != "中宫":
                kaiti_font = get_kaiti_font(bagua_font_size)  # 获取更大的楷体字体
                if kaiti_font:
                    bagua_bbox = draw.textbbox((0, 0), bagua_text, font=kaiti_font)
                    bagua_w = bagua_bbox[2] - bagua_bbox[0]
                    bagua_h = bagua_bbox[3] - bagua_bbox[1]
                    
                    bagua_x = center_x - bagua_w / 2
                    bagua_y = center_y - bagua_h / 2 - 25  # 在方位名称上方
                    
                    # 绘制宫位名称（红色楷体）
                    draw.text((bagua_x + 1, bagua_y + 1), bagua_text, font=kaiti_font, fill=(255, 255, 255, 180))
                    draw.text((bagua_x, bagua_y), bagua_text, font=kaiti_font, fill=(255, 0, 0, 255))
            
            # 计算方位名称文字位置 - 根据实际方位放在户型图外围
            if direction != "中":  # 只有"中"保留在格子内，其他方位放在外面
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 根据实际方位和north_angle确定文字在户型图外围的位置
                margin = 15  # 文字与整个户型图边界的距离
                
                # 获取整个九宫格的边界
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                # 根据方位确定文字在户型图外围的实际位置
                if direction == "北":
                    # 北方在户型图的实际北方向外侧
                    if north_angle == 90:  # 上方是北
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 270:  # 下方是北
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                    elif north_angle == 0:  # 右方是北
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 180:  # 左方是北
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                        
                elif direction == "南":
                    # 南方在户型图的实际南方向外侧
                    if north_angle == 90:  # 下方是南
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                    elif north_angle == 270:  # 上方是南
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 0:  # 左方是南
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 180:  # 右方是南
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                        
                elif direction == "东":
                    # 东方在户型图的实际东方向外侧
                    if north_angle == 90:  # 右方是东
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 270:  # 左方是东
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 0:  # 上方是东
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 180:  # 下方是东
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                        
                elif direction == "西":
                    # 西方在户型图的实际西方向外侧
                    if north_angle == 90:  # 左方是西
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 270:  # 右方是西
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 0:  # 下方是西
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                    elif north_angle == 180:  # 上方是西
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                        
                else:
                    # 对角方位的处理（东北、西北、东南、西南）
                    # 暂时使用九宫格位置，稍后可以优化
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
                        
                # 绘制方位名称（无背景框，使用阴影效果增强可读性）
                # 先绘制白色阴影
                draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
                # 再绘制黑色主文字
                draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
            else:
                # "中"仍然绘制在格子中心
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                text_x = center_x - text_w / 2
                text_y = center_y - text_h / 2 - 5
                
                # 绘制方位名称
                draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
                draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
        
        # 如果缺角，显示缺角率信息（透明背景）
        if is_missing and small_font:
            missing_rate = 1 - coverage
            coverage_text = f"缺角率: {missing_rate:.2f}"
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


def draw_luoshu_grid_only(image, polygon=None, overlay_alpha=0.7, original_image_path=None, north_angle=90):
    """在图像上绘制九宫格（仅显示方位，不显示八星），基于房间边界"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(light_image)
    
    # 创建透明overlay用于绘制九宫格
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 获取房屋边界 - 基于房间分布
    if polygon:
        min_x, min_y, max_x, max_y = get_polygon_bounds(polygon)
        print(f"使用房间边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    else:
        # 如果没有多边形，使用整个图像
        min_x, min_y, max_x, max_y = 0, 0, w, h
        print(f"使用整个图像边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    
    # 九宫格尺寸 - 基于确定的边界
    house_w = max_x - min_x
    house_h = max_y - min_y
    grid_w = house_w / 3
    grid_h = house_h / 3
    
    directions = get_luoshu_grid_positions(north_angle)
    
    # 获取字体
    font_size = min(int(house_w), int(house_h)) // 18
    font = get_chinese_font(max(16, font_size))
    bagua_font_size = max(20, font_size + 4)  # 宫位字体更大
    
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
        
        # 绘制虚线边框线
        draw_dashed_rectangle(draw, [x1, y1, x2, y2], (0, 0, 0, 255), width=1)
        
        # 计算九宫格区域中心
        center_x = x1 + grid_w / 2
        center_y = y1 + grid_h / 2
        
        # 方位名称和宫位名称
        direction_text = direction
        bagua_text = DIRECTION_TO_BAGUA.get(direction, "")
        
        # 绘制方位文字（透明背景）
        if font:
            # 先绘制宫位名称（八卦）- 使用楷体和红色
            if bagua_text and bagua_text != "中宫":
                kaiti_font = get_kaiti_font(bagua_font_size)  # 获取更大的楷体字体
                if kaiti_font:
                    bagua_bbox = draw.textbbox((0, 0), bagua_text, font=kaiti_font)
                    bagua_w = bagua_bbox[2] - bagua_bbox[0]
                    bagua_h = bagua_bbox[3] - bagua_bbox[1]
                    
                    bagua_x = center_x - bagua_w / 2
                    bagua_y = center_y - bagua_h / 2 - 15  # 在方位名称上方
                    
                    # 绘制宫位名称（红色楷体）
                    draw.text((bagua_x + 1, bagua_y + 1), bagua_text, font=kaiti_font, fill=(255, 255, 255, 180))
                    draw.text((bagua_x, bagua_y), bagua_text, font=kaiti_font, fill=(255, 0, 0, 255))
            
            # 计算方位名称文字位置 - 根据实际方位放在户型图外围
            if direction != "中":  # 只有"中"保留在格子内，其他方位放在外面
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 根据实际方位和north_angle确定文字在户型图外围的位置
                margin = 12  # 文字与整个户型图边界的距离
                
                # 获取整个九宫格的边界
                grid_min_x, grid_min_y = min_x, min_y
                grid_max_x, grid_max_y = max_x, max_y
                
                # 根据方位确定文字在户型图外围的实际位置
                if direction == "北":
                    # 北方在户型图的实际北方向外侧
                    if north_angle == 90:  # 上方是北
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 270:  # 下方是北
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                    elif north_angle == 0:  # 右方是北
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 180:  # 左方是北
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                        
                elif direction == "南":
                    # 南方在户型图的实际南方向外侧
                    if north_angle == 90:  # 下方是南
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                    elif north_angle == 270:  # 上方是南
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 0:  # 左方是南
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 180:  # 右方是南
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                        
                elif direction == "东":
                    # 东方在户型图的实际东方向外侧
                    if north_angle == 90:  # 右方是东
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 270:  # 左方是东
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 0:  # 上方是东
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 180:  # 下方是东
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                        
                elif direction == "西":
                    # 西方在户型图的实际西方向外侧
                    if north_angle == 90:  # 左方是西
                        text_x = grid_min_x - text_w - margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 270:  # 右方是西
                        text_x = grid_max_x + margin
                        text_y = (grid_min_y + grid_max_y) / 2 - text_h / 2
                    elif north_angle == 0:  # 下方是西
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_max_y + margin
                    elif north_angle == 180:  # 上方是西
                        text_x = (grid_min_x + grid_max_x) / 2 - text_w / 2
                        text_y = grid_min_y - text_h - margin
                        
                else:
                    # 对角方位的处理（东北、西北、东南、西南）
                    # 暂时使用九宫格位置，稍后可以优化
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
                        
                # 绘制方位名称（无背景框，使用阴影效果增强可读性）
                # 先绘制白色阴影
                draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
                # 再绘制黑色主文字
                draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
            else:
                # "中"仍然绘制在格子中心
                bbox = draw.textbbox((0, 0), direction_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                text_x = center_x - text_w / 2
                text_y = center_y - text_h / 2
                
                # 绘制方位名称
                draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
                draw.text((text_x, text_y), direction_text, font=font, fill=(0, 0, 0, 255))
    
    # 将透明overlay合成到原图上
    pil_image = pil_image.convert('RGBA')
    result = Image.alpha_composite(pil_image, overlay)
    
    # 转换回OpenCV格式
    return pil_to_cv2(result.convert('RGB'))

def draw_bazhai_circle(image, direction_stars_mapping, polygon=None, rooms_data=None, house_orientation=None, overlay_alpha=0.7, north_angle=90):
    """在图像上绘制八宅八星圆形图，基于户型图的最小外接圆"""
    h, w = image.shape[:2]
    
    # 将底图变为浅色系
    light_image = cv2.addWeighted(image, 0.6, np.full_like(image, 255), 0.4, 0)
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(light_image)
    
    # 创建透明overlay用于绘制八宅八星
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 计算户型图的最小外接圆（不需要调整坐标，因为外部已经处理了扩展）
    if polygon:
        # 使用多边形数据计算最小外接圆
        center_x, center_y, radius = get_minimum_enclosing_circle(polygon)
        print(f"使用多边形最小外接圆: 中心({center_x:.1f}, {center_y:.1f}), 半径{radius:.1f}")
        
        # 绘制房屋轮廓
        polygon_points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon_points, fill=None, outline=(100, 100, 100, 180), width=2)
    elif rooms_data:
        # 使用房间数据计算最小外接圆
        center_x, center_y, radius = get_minimum_enclosing_circle_from_rooms(rooms_data, w, h)
        print(f"使用房间数据最小外接圆: 中心({center_x:.1f}, {center_y:.1f}), 半径{radius:.1f}")
    else:
        # 回退到图像中心
        center_x = w / 2
        center_y = h / 2
        radius = min(w, h) / 4  # 稍微减小半径以留出更多空间
        print(f"使用图像中心圆: 中心({center_x:.1f}, {center_y:.1f}), 半径{radius:.1f}")
    
    colors = get_star_colors()
    
    # 确保圆形有足够大小，并为文字标签留出空间
    min_radius = min(w, h) / 5  # 最小半径
    radius = max(radius, min_radius)
    
    # 获取字体 - 调整为更小的字体
    font_size = int(radius) // 10  # 大幅减小字体大小
    direction_font = get_chinese_font(max(12, font_size))  # 方位文字用更小字体
    star_font = get_chinese_font(max(14, font_size + 2))   # 星位文字稍大一些
    
    # 八个方位的角度（根据north_angle动态调整）
    # 在PIL中，角度0度是右方（东），顺时针为正
    direction_angles = get_bazhai_direction_angles(north_angle)
    
    # 定义吉凶星位
    auspicious_stars = {"生气", "延年", "天医", "伏位"}  # 吉四星
    inauspicious_stars = {"绝命", "五鬼", "六煞", "祸害"}  # 凶四星
    
    # 绘制八个扇形区域
    for direction, angle in direction_angles.items():
        if direction == "中":  # 跳过中心
            continue
            
        # 计算扇形的起始和结束角度
        start_angle = angle - 22.5
        end_angle = angle + 22.5
        
        # 获取对应的星位和颜色
        star = direction_stars_mapping.get(direction, "未知")
        
        # 根据吉凶星位确定填充颜色
        if star in auspicious_stars:
            # 吉四星用半透明浅红色填充
            fill_color = (255, 200, 200, 100)  # 半透明浅红色
        elif star in inauspicious_stars:
            # 凶四星用半透明黄色填充
            fill_color = (255, 255, 150, 100)  # 半透明黄色
        else:
            fill_color = None
        
        # 绘制扇形区域，有颜色填充
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        if fill_color:
            draw.pieslice(bbox, start_angle, end_angle, fill=fill_color, outline=(0, 0, 0, 100), width=1)
        else:
            draw.pieslice(bbox, start_angle, end_angle, fill=None, outline=(0, 0, 0, 100), width=1)
        
        # 计算文字位置
        # 方位标签放在圆外面，但更靠近圆
        direction_radius = radius * 1.15  # 减少方位标签距离，让文字更靠近圆
        direction_angle_rad = math.radians(angle)
        direction_x = center_x + direction_radius * math.cos(direction_angle_rad)
        direction_y = center_y + direction_radius * math.sin(direction_angle_rad)
        
        # 星位标签放在圆内
        star_radius = radius * 0.7  # 稍微增加星位标签距离
        star_x = center_x + star_radius * math.cos(direction_angle_rad)
        star_y = center_y + star_radius * math.sin(direction_angle_rad)
        
        # 绘制方位文字（在圆外面）
        direction_text = direction
        star_text = f"{star}" if star != "未知" else star
        
        if direction_font:
            # 方位文字 - 在圆外面
            bbox = draw.textbbox((0, 0), direction_text, font=direction_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            label_x = direction_x - text_w//2
            label_y = direction_y - text_h//2
            
            # 直接绘制文字，黑色字体
            draw.text((label_x, label_y), direction_text, font=direction_font, fill=(0, 0, 0, 255))
        
        if star_font:
            # 星位文字 - 在圆内，吉星用红色，凶星用黑色
            bbox = draw.textbbox((0, 0), star_text, font=star_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            label_x = star_x - text_w//2
            label_y = star_y - text_h//2
            
            # 根据星位类型选择文字颜色
            if star in auspicious_stars:
                text_color = (200, 0, 0, 255)  # 吉星用红色
            else:
                text_color = (0, 0, 0, 255)    # 凶星用黑色
            
            draw.text((label_x, label_y), star_text, font=star_font, fill=text_color)
    
    # 绘制中心圆
    center_radius = radius / 4
    
    # 根据房屋朝向确定宅卦
    gua_name = "中"  # 默认值
    if house_orientation and house_orientation in HOUSE_ORIENTATION_TO_GUA:
        gua_name = HOUSE_ORIENTATION_TO_GUA[house_orientation]
        print(f"房屋朝向: {house_orientation} -> 宅卦: {gua_name}")
    
    # 不绘制中心圆的背景色，只绘制边框
    center_bbox = [center_x - center_radius, center_y - center_radius, 
                   center_x + center_radius, center_y + center_radius]
    draw.ellipse(center_bbox, fill=None, outline=(0, 0, 0, 255), width=2)
    
    # 中心文字 - 显示宅卦，使用更大更粗的字体
    center_font_size = max(24, int(radius) // 6)  # 中心字体更大
    center_font = get_chinese_font(center_font_size)
    
    if center_font:
        center_text = gua_name
        bbox = draw.textbbox((0, 0), center_text, font=center_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        label_x = center_x - text_w//2
        label_y = center_y - text_h//2
        
        # 绘制加粗效果 - 多次绘制偏移像素
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((label_x + dx, label_y + dy), center_text, font=center_font, fill=(0, 0, 0, 255))
        
        # 再次绘制主文字以增强效果
        draw.text((label_x, label_y), center_text, font=center_font, fill=(0, 0, 0, 255))
    
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

def create_combined_visualization(image, rooms_data, direction_stars_mapping, polygon=None, missing_corners=None, house_orientation=None, north_angle=90):
    """创建组合可视化图像：九宫格图 + 八宅八星圆形图，包含缺角信息，上下布局"""
    h, w = image.shape[:2]
    
    # 计算需要的额外空间
    title_height = 60
    
    # 为八宅八星图增加更大的留白，确保方位标签在圆外面不被截断
    # 方位标签在 radius * 1.3 的位置，所以需要更多空间
    padding_vertical = 150   # 大幅增加垂直留白，确保圆形完整显示
    padding_horizontal = 150  # 增加水平留白，确保圆形和方位标签都能完整显示
    
    # 扩展图像尺寸以容纳更大的留白
    extended_w = w + 2 * padding_horizontal
    extended_h = h + 2 * padding_vertical
    extended_image = np.full((extended_h, extended_w, 3), 255, dtype=np.uint8)  # 白色背景
    
    # 将原始图像放置在扩展图像的中央
    x_offset = padding_horizontal
    y_offset = padding_vertical
    extended_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    # 预先准备调整后的多边形和房间数据
    adjusted_polygon = polygon  # 默认使用原始多边形
    adjusted_rooms = rooms_data  # 默认使用原始房间数据
    
    if polygon:
        # 调整多边形坐标（同时有水平和垂直偏移）
        adjusted_polygon = [(x + x_offset, y + y_offset) for x, y in polygon]
    if rooms_data:
        # 调整房间坐标（同时有水平和垂直偏移）
        adjusted_rooms = []
        for room in rooms_data:
            adjusted_room = room.copy()
            if 'bbox' in room and room['bbox']:
                bbox = room['bbox'].copy()
                bbox['x1'] += x_offset
                bbox['x2'] += x_offset
                bbox['y1'] += y_offset
                bbox['y2'] += y_offset
                adjusted_room['bbox'] = bbox
            if 'center' in room and room['center']:
                center = room['center'].copy()
                center['x'] += x_offset
                center['y'] += y_offset
                adjusted_room['center'] = center
            adjusted_rooms.append(adjusted_room)
    
    # 创建两个分离的图像
    if missing_corners:
        luoshu_image = draw_luoshu_grid_with_missing_corners(extended_image.copy(), adjusted_rooms, adjusted_polygon, missing_corners=missing_corners, north_angle=north_angle)
    else:
        luoshu_image = draw_luoshu_grid_only(extended_image.copy(), adjusted_polygon, north_angle=north_angle)
    
    bazhai_image = draw_bazhai_circle(extended_image.copy(), direction_stars_mapping, adjusted_polygon, adjusted_rooms, house_orientation, north_angle=north_angle)
    
    # 标注房间位置到两个图像上
    luoshu_image = draw_room_positions(luoshu_image, adjusted_rooms)
    bazhai_image = draw_room_positions(bazhai_image, adjusted_rooms)
    
    # 垂直拼接两张图（上下布局）
    combined_image = np.vstack([luoshu_image, bazhai_image])
    
    # 为每个图像添加标题
    extended_h, extended_w = extended_image.shape[:2]
    title_area = np.zeros((title_height, extended_w, 3), dtype=np.uint8)
    title_area[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文标题
    pil_title = cv2_to_pil(title_area)
    draw = ImageDraw.Draw(pil_title)
    
    title_font = get_chinese_font(32)
    if title_font:
        # 上方图像标题（九宫格）
        if missing_corners:
            top_title = "九宫格缺角分析"
        else:
            top_title = "九宫格方位图"
        bbox = draw.textbbox((0, 0), top_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        center_x = extended_w // 2
        draw.text((center_x - title_w//2, 15), top_title, font=title_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    title_cv2 = pil_to_cv2(pil_title)
    
    # 添加九宫格标题
    luoshu_with_title = np.vstack([title_cv2, luoshu_image])
    
    # 为八宅八星图添加标题
    title_area2 = np.zeros((title_height, extended_w, 3), dtype=np.uint8)
    title_area2[:] = (50, 50, 50)
    
    pil_title2 = cv2_to_pil(title_area2)
    draw2 = ImageDraw.Draw(pil_title2)
    
    if title_font:
        # 下方图像标题（八宅八星）
        bottom_title = "八宅八星图"
        bbox = draw2.textbbox((0, 0), bottom_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        center_x = extended_w // 2
        draw2.text((center_x - title_w//2, 15), bottom_title, font=title_font, fill=(255, 255, 255))
    
    title_cv2_2 = pil_to_cv2(pil_title2)
    bazhai_with_title = np.vstack([title_cv2_2, bazhai_image])
    
    # 最终垂直拼接
    final_image = np.vstack([luoshu_with_title, bazhai_with_title])
    
    return final_image


def create_combined_visualization_old(image, rooms_data, direction_stars_mapping, polygon=None, north_angle=90):
    """创建组合可视化图像：九宫格图 + 八宅八星圆形图"""
    h, w = image.shape[:2]
    
    # 创建两个分离的图像
    luoshu_image = draw_luoshu_grid_only(image.copy(), polygon, north_angle=north_angle)
    bazhai_image = draw_bazhai_circle(image.copy(), direction_stars_mapping, polygon, rooms_data, north_angle=north_angle)
    
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
            missing_title = "缺角分析 (缺角率 > 10%)"
            bbox = draw.textbbox((0, 0), missing_title, font=title_font)
            title_w = bbox[2] - bbox[0]
            draw.text((w//2 - title_w//2, missing_y), missing_title, font=title_font, fill=(255, 255, 255))
        
        # 绘制缺角信息
        missing_start_y = missing_y + 35
        for i, corner in enumerate(missing_corners):
            x = start_x + (i % 6) * (col_width * 0.8)
            y = missing_start_y + (i // 6) * 25
            
            missing_rate = 1 - corner['coverage']
            text = f"{corner['direction']}方: {missing_rate:.2f}"
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
    
    # 获取图像路径和北向角度
    meta = raw_data.get('meta', {})
    original_image_path = meta.get('original_image')
    result_image_path = meta.get('result_image') or meta.get('output_image')
    north_angle = meta.get('north_angle', 90)  # 默认90度（上方为北）
    
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
        rooms_for_analysis, doc.img_w, doc.img_h, threshold=0.9, north_angle=north_angle
    )
    
    if missing_corners:
        print(f"\n缺角分析结果:")
        for corner in missing_corners:
            print(f"    {corner['direction']}方: 覆盖率 {corner['coverage']:.3f}")
    else:
        print(f"\n缺角分析结果: 无明显缺角")
    
    # 创建组合可视化图像（包含缺角信息）
    house_orientation = getattr(doc, 'house_orientation', '坐北朝南')
    final_image = create_combined_visualization(image, rooms, direction_stars_mapping, detailed_polygon, missing_corners, house_orientation, north_angle)
    
    # 去掉图例，按用户要求
    # final_image = add_legend(final_image, direction_stars_mapping, missing_corners)
    
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
        
        # 自动打开生成的图片
        import os
        import subprocess
        if os.path.exists(output_path):
            try:
                # Windows系统使用start命令打开图片
                subprocess.run(['start', '', output_path], shell=True, check=True)
                print(f"📖 已自动打开图片: {output_path}")
            except subprocess.CalledProcessError:
                print(f"⚠️ 无法自动打开图片，请手动查看: {output_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
