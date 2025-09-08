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
from typing import List, Dict, Any, Tuple, Optional

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

# 二十四山系统数据定义 - 按标准风水罗盘定义，从正北(子)开始顺时针
TWENTY_FOUR_MOUNTAINS = [
    # 北方三山 (337.5° - 22.5°) - 坎宫
    {"name": "壬", "angle": 337.5, "group": "北方", "bagua": "坎", "type": "天干", "color": (100, 149, 237)},
    {"name": "子", "angle": 0.0, "group": "北方", "bagua": "坎", "type": "地支", "color": (70, 130, 180)},      # 正北
    {"name": "癸", "angle": 22.5, "group": "北方", "bagua": "坎", "type": "天干", "color": (100, 149, 237)},
    
    # 东北三山 (22.5° - 67.5°) - 艮宫
    {"name": "丑", "angle": 37.5, "group": "东北", "bagua": "艮", "type": "地支", "color": (205, 133, 63)},
    {"name": "艮", "angle": 45.0, "group": "东北", "bagua": "艮", "type": "八卦", "color": (160, 82, 45)},      # 东北正中
    {"name": "寅", "angle": 52.5, "group": "东北", "bagua": "艮", "type": "地支", "color": (205, 133, 63)},
    
    # 东方三山 (67.5° - 112.5°) - 震宫
    {"name": "甲", "angle": 82.5, "group": "东方", "bagua": "震", "type": "天干", "color": (34, 139, 34)},
    {"name": "卯", "angle": 90.0, "group": "东方", "bagua": "震", "type": "地支", "color": (0, 128, 0)},      # 正东
    {"name": "乙", "angle": 97.5, "group": "东方", "bagua": "震", "type": "天干", "color": (34, 139, 34)},
    
    # 东南三山 (112.5° - 157.5°) - 巽宫
    {"name": "辰", "angle": 127.5, "group": "东南", "bagua": "巽", "type": "地支", "color": (72, 209, 204)},
    {"name": "巽", "angle": 135.0, "group": "东南", "bagua": "巽", "type": "八卦", "color": (0, 191, 255)},     # 东南正中
    {"name": "巳", "angle": 142.5, "group": "东南", "bagua": "巽", "type": "地支", "color": (72, 209, 204)},
    
    # 南方三山 (157.5° - 202.5°) - 离宫
    {"name": "丙", "angle": 172.5, "group": "南方", "bagua": "离", "type": "天干", "color": (220, 20, 60)},
    {"name": "午", "angle": 180.0, "group": "南方", "bagua": "离", "type": "地支", "color": (255, 0, 0)},     # 正南
    {"name": "丁", "angle": 187.5, "group": "南方", "bagua": "离", "type": "天干", "color": (220, 20, 60)},
    
    # 西南三山 (202.5° - 247.5°) - 坤宫
    {"name": "未", "angle": 217.5, "group": "西南", "bagua": "坤", "type": "地支", "color": (255, 215, 0)},
    {"name": "坤", "angle": 225.0, "group": "西南", "bagua": "坤", "type": "八卦", "color": (255, 165, 0)},     # 西南正中
    {"name": "申", "angle": 232.5, "group": "西南", "bagua": "坤", "type": "地支", "color": (255, 215, 0)},
    
    # 西方三山 (247.5° - 292.5°) - 兑宫
    {"name": "庚", "angle": 262.5, "group": "西方", "bagua": "兑", "type": "天干", "color": (192, 192, 192)},
    {"name": "酉", "angle": 270.0, "group": "西方", "bagua": "兑", "type": "地支", "color": (169, 169, 169)},   # 正西
    {"name": "辛", "angle": 277.5, "group": "西方", "bagua": "兑", "type": "天干", "color": (192, 192, 192)},
    
    # 西北三山 (292.5° - 337.5°) - 乾宫
    {"name": "戌", "angle": 307.5, "group": "西北", "bagua": "乾", "type": "地支", "color": (138, 43, 226)},
    {"name": "乾", "angle": 315.0, "group": "西北", "bagua": "乾", "type": "八卦", "color": (75, 0, 130)},      # 西北正中
    {"name": "亥", "angle": 322.5, "group": "西北", "bagua": "乾", "type": "地支", "color": (138, 43, 226)},
]


def get_bagua_from_angle(angle: float) -> str:
    """根据角度获取所属八卦宫位（使用二十四山划分）"""
    best_bagua = "未知"
    best_diff = 360.0
    for m in TWENTY_FOUR_MOUNTAINS:
        start = (m["angle"] - 7.5) % 360
        end = (m["angle"] + 7.5) % 360
        if start < end:
            if start <= angle < end:
                return m["bagua"]
        else:  # 跨越0度
            if angle >= start or angle < end:
                return m["bagua"]
        diff = min(abs(angle - m["angle"]), 360 - abs(angle - m["angle"]))
        if diff < best_diff:
            best_diff = diff
            best_bagua = m["bagua"]
    return best_bagua


def get_angle_from_grid_position(gx: int, gy: int, north_angle: int = 0) -> Optional[float]:
    """计算九宫格位置相对于北方的角度"""
    dx = gx - 1
    dy = gy - 1
    if dx == 0 and dy == 0:
        return None
    angle = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    angle = (angle - north_angle + 360.0) % 360.0
    return angle


def get_direction_from_grid_position(gx: int, gy: int, north_angle: int = 0) -> str:
    """根据九宫格位置获取方位名称"""
    angle = get_angle_from_grid_position(gx, gy, north_angle)
    if angle is None:
        return "中"
    direction_names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    idx = int(((angle + 22.5) % 360) / 45)
    return direction_names[idx]


def get_bagua_from_direction(direction: str) -> str:
    """根据方位名称获取八卦宫位"""
    direction_to_bagua = {
        "北": "坎",
        "东北": "艮", 
        "东": "震",
        "东南": "巽",
        "南": "离",
        "西南": "坤",
        "西": "兑",
        "西北": "乾",
        "中": "中"
    }
    return direction_to_bagua.get(direction, "中")

def get_bagua_from_grid_position(gx: int, gy: int, north_angle: int = 0) -> str:
    """根据九宫格位置获取八卦宫位"""
    angle = get_angle_from_grid_position(gx, gy, north_angle)
    if angle is None:
        return "中宫"
    return get_bagua_from_angle(angle)

def load_json_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_direction_label_position(direction: str, grid_bounds: tuple, text_w: int, text_h: int, north_angle: int, margin: int = 15):
    """计算方向标签在九宫格外部的位置
    
    Args:
        direction: 方位名称 (北、南、东、西、东北、西北、东南、西南)
        grid_bounds: 九宫格边界 (min_x, min_y, max_x, max_y)
        text_w, text_h: 文字宽度和高度
        north_angle: 北方角度
        margin: 边距
        
    Returns:
        (text_x, text_y): 文字位置坐标
    """
    min_x, min_y, max_x, max_y = grid_bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # 计算九宫格的半径（用于确定外部距离）
    grid_radius = max((max_x - min_x) / 2, (max_y - min_y) / 2)
    
    # 方位到角度的基础映射（以北为0度，顺时针）
    direction_angles = {
        "北": 0,
        "东北": 45,
        "东": 90,
        "东南": 135,
        "南": 180,
        "西南": 225,
        "西": 270,
        "西北": 315
    }
    
    if direction not in direction_angles:
        # 中心位置或其他
        return center_x - text_w / 2, center_y - text_h / 2
    
    # 计算实际角度（考虑north_angle偏移）
    # north_angle为正表示北方顺时针偏移，所以所有方位都要顺时针旋转
    base_angle = direction_angles[direction]
    actual_angle = (base_angle + north_angle) % 360
    
    # 转换为弧度
    angle_rad = math.radians(actual_angle)
    
    # 计算距离九宫格中心的距离（确保在外部）
    distance = grid_radius + margin + max(text_w, text_h) / 2
    
    # 计算标签位置
    dx = distance * math.sin(angle_rad)  # sin对应x方向（图像坐标系）
    dy = -distance * math.cos(angle_rad)  # -cos对应y方向（向上为负）
    
    # 计算最终位置（考虑文字尺寸的偏移）
    text_x = center_x + dx - text_w / 2
    text_y = center_y + dy - text_h / 2
    
    return text_x, text_y


def create_polygon_from_rooms(rooms: List[Dict[str, Any]], shrink_balcony: bool = False) -> List[tuple]:
    """从房间数据创建更精确的外轮廓多边形

    Parameters
    ----------
    rooms : list of dict
        房间数据，每个dict包含bbox信息。
    shrink_balcony : bool, optional
        是否对阳台做收缩处理。True时将阳台较短一边缩小为一半，
        False时保持原始尺寸，默认False。
    """
    if not rooms:
        return []

    # 收集所有房间的边界框，必要时对阳台宽度按一半计算
    boxes = []
    for room in rooms:
        bbox = room.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            room_type = str(room.get("type", ""))
            # 阳台：根据参数决定是否收缩
            if shrink_balcony and room_type == "阳台":
                w = x2 - x1
                h = y2 - y1
                if abs(w) <= abs(h):
                    x2 = x1 + w * 0.5
                else:
                    y1 = y2 - h * 0.5
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

def get_direction_from_point(cx: float, cy: float, ox: float, oy: float, north_angle: int = 0) -> str:
    """Convert a point to compass direction considering north angle.
    
    Args:
        cx, cy: Point coordinates
        ox, oy: Origin/center coordinates  
        north_angle: North direction angle (0=North, 90=East, 180=South, 270=West)
        
    Returns:
        Direction name in Chinese
    """
    DIRECTION_NAMES = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    
    dx = cx - ox
    dy = cy - oy
    # 在标准数学坐标系中：0°=东(右)，90°=北(上)
    # 转换为指北针系统：0°=北，90°=东，180°=南，270°=西
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0  # 数学坐标系角度
    angle = (90 - angle + 360.0) % 360.0  # 转换为指北针角度：0°=北
    angle = (angle - north_angle + 360.0) % 360.0  # 应用north_angle偏移
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]


def convert_compass_to_pil_angle(compass_angle: float) -> float:
    """将罗盘角度转换为PIL绘图角度
    
    罗盘角度: 北=0°, 顺时针递增
    PIL角度: 东=0°, 顺时针递增
    
    Args:
        compass_angle: 罗盘角度（北=0°，顺时针）
    Returns:
        PIL角度（东=0°，顺时针）
    """
    # 转换公式：
    # 罗盘北(0°) -> PIL北(270°)
    # 罗盘东(90°) -> PIL东(0°)  
    # 罗盘南(180°) -> PIL南(90°)
    # 罗盘西(270°) -> PIL西(180°)
    pil_angle = (compass_angle + 270) % 360
    return pil_angle


def get_bazhai_direction_angles(north_angle: int = 0) -> Dict[str, float]:
    """根据north_angle返回八宅八星方位角度映射
    
    Args:
        north_angle: 北方角度 (0=北, 90=东, 180=南, 270=西)
    Returns:
        方位到角度的映射字典
    """
    # 基础角度偏移量 = north_angle (统一使用罗盘坐标系)
    angle_offset = north_angle
    
    # 八个方位的基础角度（统一使用罗盘坐标系：北为0度，顺时针）
    base_angles = {
        "北": 0,      # 上方
        "东北": 45,   # 右上
        "东": 90,     # 右方
        "东南": 135,  # 右下
        "南": 180,    # 下方
        "西南": 225,  # 左下
        "西": 270,    # 左方
        "西北": 315   # 左上
    }
    
    # 应用角度偏移
    adjusted_angles = {}
    for direction, base_angle in base_angles.items():
        adjusted_angles[direction] = (base_angle + angle_offset) % 360
    
    return adjusted_angles


def get_luoshu_grid_positions(north_angle: int = 0) -> Dict[str, Tuple[int, int]]:
    """返回九宫格方位映射，根据north_angle动态调整
    
    Args:
        north_angle: 北方角度 (0=北, 90=东, 180=南, 270=西)
    """
    # 根据north_angle确定上方是什么方向
    if north_angle == 0:
        # 标准情况：上方是北方(0°)
        return {
            "西北": (0, 0), "北": (1, 0), "东北": (2, 0),
            "西": (0, 1),   "中": (1, 1), "东": (2, 1),
            "西南": (0, 2), "南": (1, 2), "东南": (2, 2)
        }
    elif north_angle == 90:
        # 上方是东方(90°)
        return {
            "东北": (0, 0), "东": (1, 0), "东南": (2, 0),
            "北": (0, 1),   "中": (1, 1), "南": (2, 1),
            "西北": (0, 2), "西": (1, 2), "西南": (2, 2)
        }
    elif north_angle == 180:
        # 上方是南方(180°)
        return {
            "东南": (0, 0), "南": (1, 0), "西南": (2, 0),
            "东": (0, 1),   "中": (1, 1), "西": (2, 1),
            "东北": (0, 2), "北": (1, 2), "西北": (2, 2)
        }
    elif north_angle == 270:
        # 上方是西方(270°)
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


def compute_luoshu_grid_positions_by_overlap(center_x, center_y, radius, north_angle=0, grid_bounds=None):
    """基于与二十四山图扇形区域重叠面积计算九宫格方位映射
    
    Args:
        center_x, center_y: 二十四山图的中心坐标
        radius: 二十四山图的半径
        north_angle: 北方角度偏移
        grid_bounds: 九宫格边界 (min_x, min_y, max_x, max_y)
    
    Returns:
        Dict[str, Tuple[int, int]]: 方位名称到九宫格位置的映射
    """
    import math
    
    try:
        from shapely.geometry import Polygon, Point
        from shapely.ops import unary_union
        HAS_SHAPELY = True
    except ImportError:
        print("⚠️ Shapely库未安装，将使用几何中心方法")
        HAS_SHAPELY = False
    
    # 如果没有提供grid_bounds，使用默认的相对于中心的九宫格
    if grid_bounds is None:
        grid_size = radius * 1.5  # 九宫格大小相对于圆的半径
        min_x = center_x - grid_size / 2
        min_y = center_y - grid_size / 2
        max_x = center_x + grid_size / 2
        max_y = center_y + grid_size / 2
    else:
        min_x, min_y, max_x, max_y = grid_bounds
    
    # 计算九宫格每个格子的边界
    grid_w = (max_x - min_x) / 3
    grid_h = (max_y - min_y) / 3
    
    # 定义八个方位的角度范围（以北为0度，顺时针）
    # 注意：这里的角度需要与二十四山图的实际扇形对应
    directions = [
        ("北", 337.5, 22.5),    # 北方：337.5° 到 22.5°（跨越0度）
        ("东北", 22.5, 67.5),   # 东北：22.5° 到 67.5°
        ("东", 67.5, 112.5),    # 东方：67.5° 到 112.5°  
        ("东南", 112.5, 157.5), # 东南：112.5° 到 157.5°
        ("南", 157.5, 202.5),   # 南方：157.5° 到 202.5°
        ("西南", 202.5, 247.5), # 西南：202.5° 到 247.5°
        ("西", 247.5, 292.5),   # 西方：247.5° 到 292.5°
        ("西北", 292.5, 337.5)  # 西北：292.5° 到 337.5°
    ]
    
    if not HAS_SHAPELY:
        # 降级到基于几何中心的方法
        print(f"\n🔍 使用几何中心方法计算九宫格方位 (north_angle={north_angle}°):")
        mapping = {}
        
        for row in range(3):
            for col in range(3):
                if col == 1 and row == 1:
                    mapping["中"] = (col, row)
                    continue
                
                # 计算格子中心
                grid_center_x = min_x + (col + 0.5) * grid_w
                grid_center_y = min_y + (row + 0.5) * grid_h
                
                # 计算相对于二十四山图中心的角度
                dx = grid_center_x - center_x
                dy = grid_center_y - center_y
                
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    direction = "中"
                else:
                    # 计算角度（以北为0度，顺时针）
                    angle = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
                    angle = (angle - north_angle + 360.0) % 360.0
                    
                    # 找到最接近的方位
                    best_direction = "北"
                    min_diff = float('inf')
                    
                    for direction_name, start_angle, end_angle in directions:
                        # 计算角度到方位中心的距离
                        if start_angle > end_angle:  # 跨越0度的情况
                            center_angle = ((start_angle + end_angle + 360) / 2) % 360
                            # 处理跨越0度的角度差
                            if angle >= start_angle or angle <= end_angle:
                                if angle >= start_angle:
                                    diff = min(abs(angle - center_angle), abs(angle - center_angle + 360))
                                else:
                                    diff = min(abs(angle - center_angle), abs(angle - center_angle - 360))
                            else:
                                diff = min(abs(angle - start_angle), abs(angle - end_angle))
                        else:
                            center_angle = (start_angle + end_angle) / 2
                            diff = abs(angle - center_angle)
                        
                        if diff < min_diff:
                            min_diff = diff
                            best_direction = direction_name
                    
                    direction = best_direction
                
                print(f"  格子({col},{row}) 中心({grid_center_x:.0f},{grid_center_y:.0f}) 角度{angle:.1f}° → {direction}")
                
                # 处理冲突：如果方位已存在，选择距离中心更近的格子
                if direction in mapping and direction != "中":
                    old_col, old_row = mapping[direction]
                    old_center_x = min_x + (old_col + 0.5) * grid_w
                    old_center_y = min_y + (old_row + 0.5) * grid_h
                    old_distance = math.sqrt((old_center_x - center_x)**2 + (old_center_y - center_y)**2)
                    new_distance = math.sqrt((grid_center_x - center_x)**2 + (grid_center_y - center_y)**2)
                    
                    if new_distance < old_distance:
                        print(f"    ⚠️ 方位冲突: {direction}方 - 保留更接近中心的格子({col},{row})")
                        mapping[direction] = (col, row)
                    else:
                        print(f"    ⚠️ 方位冲突: {direction}方 - 保留原有格子({old_col},{old_row})")
                else:
                    mapping[direction] = (col, row)
        
        return mapping
    
    # 使用Shapely的精确重叠面积计算
    def create_sector_polygon(center_x, center_y, radius, start_angle, end_angle, north_angle):
        """创建扇形多边形"""
        points = [(center_x, center_y)]  # 从中心开始
        
        # 应用北角偏移
        start_angle = (start_angle + north_angle) % 360
        end_angle = (end_angle + north_angle) % 360
        
        # 处理跨越0度的情况
        if start_angle > end_angle:
            # 分两段处理
            angles = list(range(int(start_angle), 360, 5)) + list(range(0, int(end_angle) + 1, 5))
        else:
            angles = list(range(int(start_angle), int(end_angle) + 1, 5))
        
        for angle in angles:
            rad = math.radians(angle)
            x = center_x + radius * math.sin(rad)
            y = center_y - radius * math.cos(rad)
            points.append((x, y))
        
        return Polygon(points)
    
    # 计算每个九宫格格子与各方位扇形的重叠面积
    mapping = {}
    overlap_data = {}  # 存储重叠面积数据用于冲突解决
    
    print(f"\n🔍 基于重叠面积计算九宫格方位 (north_angle={north_angle}°):")
    
    # 第一阶段：计算所有格子与所有方位的重叠面积
    for row in range(3):
        for col in range(3):
            if col == 1 and row == 1:
                mapping["中"] = (col, row)
                continue  # 跳过中心格子
            
            # 创建九宫格格子的多边形
            grid_x1 = min_x + col * grid_w
            grid_y1 = min_y + row * grid_h
            grid_x2 = min_x + (col + 1) * grid_w
            grid_y2 = min_y + (row + 1) * grid_h
            
            grid_polygon = Polygon([
                (grid_x1, grid_y1), (grid_x2, grid_y1),
                (grid_x2, grid_y2), (grid_x1, grid_y2)
            ])
            
            overlap_data[(col, row)] = {}
            
            print(f"  格子({col},{row}) 区域: ({grid_x1:.0f},{grid_y1:.0f})-({grid_x2:.0f},{grid_y2:.0f})")
            
            # 计算与每个方位扇形的重叠面积
            for direction_name, start_angle, end_angle in directions:
                sector_polygon = create_sector_polygon(
                    center_x, center_y, radius, start_angle, end_angle, north_angle
                )
                
                try:
                    # 计算重叠面积
                    intersection = grid_polygon.intersection(sector_polygon)
                    overlap_area = intersection.area
                    overlap_data[(col, row)][direction_name] = overlap_area
                    
                    print(f"    与{direction_name}方重叠面积: {overlap_area:.0f}")
                        
                except Exception as e:
                    print(f"    计算与{direction_name}方重叠时出错: {e}")
                    overlap_data[(col, row)][direction_name] = 0
                    continue
    
    # 第二阶段：为每个格子分配最佳方位，同时处理冲突
    for row in range(3):
        for col in range(3):
            if col == 1 and row == 1:
                continue  # 中心格子已处理
            
            # 找到该格子重叠面积最大的方位
            max_overlap_area = 0
            best_direction = "未知"
            
            for direction_name in [d[0] for d in directions]:
                overlap_area = overlap_data[(col, row)].get(direction_name, 0)
                if overlap_area > max_overlap_area:
                    max_overlap_area = overlap_area
                    best_direction = direction_name
            
            print(f"  格子({col},{row}) → 最佳方位: {best_direction}方 (重叠面积{max_overlap_area:.0f})")
            
            # 检查方位冲突并解决
            if best_direction in mapping and best_direction != "中":
                old_col, old_row = mapping[best_direction]
                old_overlap = overlap_data[(old_col, old_row)].get(best_direction, 0)
                
                print(f"    ⚠️ 方位冲突: {best_direction}方")
                print(f"       旧格子({old_col},{old_row}) 重叠面积: {old_overlap:.0f}")
                print(f"       新格子({col},{row}) 重叠面积: {max_overlap_area:.0f}")
                
                if max_overlap_area > old_overlap:
                    print(f"       → 保留新格子({col},{row})")
                    
                    # 为旧格子重新分配次优方位
                    old_second_best_direction = "未分配"
                    old_second_best_area = 0
                    for direction_name in [d[0] for d in directions]:
                        if direction_name == best_direction or direction_name in mapping:
                            continue
                        area = overlap_data[(old_col, old_row)].get(direction_name, 0)
                        if area > old_second_best_area:
                            old_second_best_area = area
                            old_second_best_direction = direction_name
                    
                    if old_second_best_direction != "未分配":
                        mapping[old_second_best_direction] = (old_col, old_row)
                        print(f"       → 旧格子({old_col},{old_row}) 重新分配到: {old_second_best_direction}方")
                    
                    mapping[best_direction] = (col, row)
                else:
                    print(f"       → 保留旧格子({old_col},{old_row})")
                    
                    # 为当前格子寻找次优方位
                    second_best_direction = "未分配"
                    second_best_area = 0
                    for direction_name in [d[0] for d in directions]:
                        if direction_name == best_direction or direction_name in mapping:
                            continue
                        area = overlap_data[(col, row)].get(direction_name, 0)
                        if area > second_best_area:
                            second_best_area = area
                            second_best_direction = direction_name
                    
                    if second_best_direction != "未分配":
                        mapping[second_best_direction] = (col, row)
                        print(f"       → 当前格子({col},{row}) 分配到次优方位: {second_best_direction}方")
            else:
                mapping[best_direction] = (col, row)
    
    print(f"\n最终映射结果:")
    for direction, position in mapping.items():
        print(f"  {direction} → 格子{position}")
    
    return mapping


def compute_luoshu_grid_positions(north_angle: int = 0) -> Dict[str, Tuple[int, int]]:
    """更通用的九宫格方位映射，支持任意 north_angle。

    通过每个格子中心相对九宫中心的方向，结合 north_angle，
    反推得到 direction -> (col, row) 的映射。
    """
    mapping: Dict[str, Tuple[int, int]] = {}
    for row in range(3):
        for col in range(3):
            direction = get_direction_from_grid_position(col, row, north_angle)
            mapping[direction] = (col, row)
    return mapping


def cv2_to_pil(cv2_image):
    """将OpenCV图像转换为PIL图像"""
    if cv2_image is None or cv2_image.size == 0:
        raise ValueError("传入的OpenCV图像为空或无效")
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
    """返回八星对应的BGR颜色，按吉凶区分"""
    colors = {"中": (128, 128, 128)}
    for star, (nature, _) in STAR_INFO.items():
        if nature == "吉":
            colors[star] = (0, 255, 255)   # 黄色
        elif nature == "凶":
            colors[star] = (0, 0, 255)     # 红色
        else:
            colors[star] = (128, 128, 128)
    return colors

def draw_luoshu_grid_with_missing_corners(image, rooms_data, polygon=None, overlay_alpha=0.7, missing_corners=None, original_image_path=None, north_angle=0, use_overlap_method=True):
    """在图像上绘制九宫格，显示缺角信息，支持动态朝向
    
    Args:
        use_overlap_method: 是否使用基于重叠面积的方位计算方法
    """
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
            temp_polygon = create_polygon_from_rooms(rooms_data, shrink_balcony=True)
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
    
    # 计算二十四山图的中心和半径（用于重叠面积计算）
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    radius = min(house_w, house_h) / 2 * 1.2  # 稍大于房屋边界的半径
    
    # 选择方位计算方法
    if use_overlap_method:
        print(f"🎯 使用基于重叠面积的方位计算方法")
        print(f"   二十四山图中心: ({center_x:.0f}, {center_y:.0f})")
        print(f"   二十四山图半径: {radius:.0f}")
        directions = compute_luoshu_grid_positions_by_overlap(
            center_x, center_y, radius, north_angle, 
            grid_bounds=(min_x, min_y, max_x, max_y)
        )
    else:
        print(f"🎯 使用传统几何角度计算方法")
        directions = compute_luoshu_grid_positions(north_angle)
    
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
        # 根据方位名称获取八卦宫位（使用新的映射结果）
        bagua_text = get_bagua_from_direction(direction)
        
        # 新通用绘制：严格按 north_angle 放置外圈方位标签；中宫居中
        if font:
            # 宫位（红色楷体）
            if bagua_text and bagua_text != "中":
                kaiti_font = get_kaiti_font(bagua_font_size)
                if kaiti_font:
                    bagua_bbox = draw.textbbox((0, 0), bagua_text, font=kaiti_font)
                    bagua_w = bagua_bbox[2] - bagua_bbox[0]
                    bagua_h = bagua_bbox[3] - bagua_bbox[1]
                    bagua_x = center_x - bagua_w / 2
                    bagua_y = center_y - bagua_h / 2 - 15
                    draw.text((bagua_x + 1, bagua_y + 1), bagua_text, font=kaiti_font, fill=(255, 255, 255, 180))
                    draw.text((bagua_x, bagua_y), bagua_text, font=kaiti_font, fill=(255, 0, 0, 255))

            # 方位标签位置（中宫例外，其他按通用算法放在外圈）
            bbox = draw.textbbox((0, 0), direction_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if direction_text == "中":
                text_x = center_x - text_w / 2
                text_y = center_y - text_h / 2
            else:
                text_x, text_y = calculate_direction_label_position(
                    direction_text,
                    (min_x, min_y, max_x, max_y),
                    text_w, text_h,
                    north_angle,
                    margin=12,
                )
            # 绘制（阴影 + 主文字）
            draw.text((text_x + 1, text_y + 1), direction_text, font=font, fill=(255, 255, 255, 180))
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
                missing_rate_y = center_y + 30  # 在中心下方
                
                # 绘制缺角率信息（红色）
                draw.text((missing_rate_x + 1, missing_rate_y + 1), missing_rate_text, font=small_font, fill=(255, 255, 255, 180))
                draw.text((missing_rate_x, missing_rate_y), missing_rate_text, font=small_font, fill=(255, 0, 0, 255))
            
            # 跳过旧的按象限逻辑
            continue
        
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
            
            # 使用通用函数计算方向标签位置
            bbox = draw.textbbox((0, 0), direction_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            if direction != "中":
                # 对于所有方向（除了中心），使用通用算法计算外部位置
                text_x, text_y = calculate_direction_label_position(
                    direction, 
                    (min_x, min_y, max_x, max_y), 
                    text_w, text_h, 
                    north_angle, 
                    margin=15
                )
            else:
                # 中心位置
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
            temp_polygon = create_polygon_from_rooms(rooms_data, shrink_balcony=True)
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
    
    directions = compute_luoshu_grid_positions(north_angle)
    
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
        bagua_text = get_bagua_from_direction(direction)
        
        # 绘制方位文字（透明背景）
        if font:
            # 先绘制宫位名称（八卦）- 使用楷体和红色
            if bagua_text and bagua_text != "中":
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


def draw_luoshu_grid_only(image, polygon=None, overlay_alpha=0.7, original_image_path=None, north_angle=0):
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
    
    # 使用基于重叠面积的方位计算方法
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    radius = min(house_w, house_h) / 2 * 1.2
    directions = compute_luoshu_grid_positions_by_overlap(
        center_x, center_y, radius, north_angle, 
        grid_bounds=(min_x, min_y, max_x, max_y)
    )
    
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
        bagua_text = get_bagua_from_direction(direction)
        
        # 绘制方位文字（透明背景）
        if font:
            # 先绘制宫位名称（八卦）- 使用楷体和红色
            if bagua_text and bagua_text != "中":
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

def draw_bazhai_circle(image, direction_stars_mapping, polygon=None, rooms_data=None, house_orientation=None, overlay_alpha=0.7, north_angle=0):
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
    direction_angles = get_bazhai_direction_angles(north_angle)

    # 绘制八个扇形区域并收集文字坐标
    star_infos = []
    for direction, angle in direction_angles.items():
        if direction == "中":  # 跳过中心
            continue

        # 获取对应的星位（必要时用朝向固定表补全）
        star = direction_stars_mapping.get(direction)
        if not star:
            try:
                from fengshui.bazhai_eightstars import HOUSE_DIRECTION_STARS
                if house_orientation and house_orientation in HOUSE_DIRECTION_STARS:
                    star = HOUSE_DIRECTION_STARS[house_orientation].get(direction)
            except Exception:
                pass
        if not star:
            star = "未知"

        # 计算扇形的起始和结束角度
        # 先计算罗盘角度的扇形范围
        compass_start_angle = angle - 22.5
        compass_end_angle = angle + 22.5
        
        # 转换为PIL绘图角度
        pil_start_angle = convert_compass_to_pil_angle(compass_start_angle)
        pil_end_angle = convert_compass_to_pil_angle(compass_end_angle)
        
        # 处理跨越0°的扇形（如315°到45°）
        if pil_end_angle < pil_start_angle:
            # 分成两个扇形绘制：[start, 360°] 和 [0°, end]
            pass  # 简化处理：大部分情况下不会跨越
        star = star.strip()
        nature = STAR_INFO.get(star, ("", ""))[0]

        # 根据吉凶星位确定填充颜色，透明度20%
        # PIL使用RGB格式进行绘制
        alpha = int(255 * 0.2)
        if nature == "吉":
            fill_color = (255, 0, 0, alpha)    # RGB格式的红色
            color_desc = "浅透明红色"
        elif nature == "凶":
            fill_color = (255, 255, 0, alpha)  # RGB格式的黄色
            color_desc = "浅透明黄色"
        else:
            fill_color = None
            color_desc = "无"


        # 绘制扇形区域，有颜色填充
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        if fill_color:
            # 先绘制填充色的扇形
            draw.pieslice(bbox, pil_start_angle, pil_end_angle, fill=fill_color, outline=None)
            #print(f'======{star}、{bbox}、{pil_start_angle}、{pil_end_angle}、{fill_color}');
            # 再绘制黑色边框
            draw.pieslice(bbox, pil_start_angle, pil_end_angle, fill=None, outline=(0, 0, 0, 200), width=2)
        else:
            # 没有填充色时，只绘制边框
            draw.pieslice(bbox, pil_start_angle, pil_end_angle, fill=None, outline=(0, 0, 0, 200), width=2)

        # 计算文字位置（统一使用罗盘坐标系的转换公式）
        # 方位标签放在圆外面，但更靠近圆
        direction_radius = radius * 1.15  # 减少方位标签距离，让文字更靠近圆
        direction_angle_rad = math.radians(angle)
        direction_x = center_x + direction_radius * math.sin(direction_angle_rad)
        direction_y = center_y - direction_radius * math.cos(direction_angle_rad)

        # 星位标签放在圆内，根据方位动态调整距离以避免靠近边缘被截断
        star_radius_factor = 0.7
        if direction == "北":
            star_radius_factor = 0.6
        elif direction in ("东北", "西北"):
            star_radius_factor = 0.65
        star_radius = radius * star_radius_factor
        star_x = center_x + star_radius * math.sin(direction_angle_rad)
        star_y = center_y - star_radius * math.cos(direction_angle_rad)
        star_infos.append((direction, star, nature, direction_x, direction_y, star_x, star_y, color_desc))

    # 绘制文字并打印坐标
    for direction, star, nature, direction_x, direction_y, star_x, star_y, color_desc in star_infos:
        direction_text = direction
        star_text = f"{star}" if star != "未知" else star
        print(f"{direction_text} 星位 '{star_text}' 方位坐标({direction_x:.1f}, {direction_y:.1f}), 星位坐标({star_x:.1f}, {star_y:.1f}), 扇形颜色{color_desc}")


        if direction_font:
            # 方位文字 - 在圆外面
            bbox = draw.textbbox((0, 0), direction_text, font=direction_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            label_x = direction_x - text_w//2
            label_y = direction_y - text_h//2

            # 防止文字超出边界
            label_x = max(0, min(label_x, w - text_w))
            label_y = max(0, min(label_y, h - text_h))

            # 直接绘制文字，黑色字体
            draw.text((label_x, label_y), direction_text, font=direction_font, fill=(0, 0, 0, 255))

        if star_font:
            # 星位文字 - 在圆内，根据吉凶选择文字颜色
            bbox = draw.textbbox((0, 0), star_text, font=star_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            label_x = star_x - text_w//2
            label_y = star_y - text_h//2

            # 防止文字超出边界并留出边距，针对北向额外增加安全距离
            padding_x = padding_y = 5
            if direction == "北":
                padding_y = 20
            elif direction in ("东北", "西北"):
                padding_y = 20
                padding_x = 20
            label_x = max(padding_x, min(label_x, w - text_w - padding_x))
            label_y = max(padding_y, min(label_y, h - text_h - padding_y))

            # 根据星位类型选择文字颜色
            if nature == "吉":
                text_color = (200, 0, 0, 255)  # 吉星用红色
            elif nature == "凶":
                text_color = (0, 0, 0, 255)    # 凶星用黑色
            else:
                text_color = (0, 0, 0, 255)

            # 在星位文字下绘制半透明白底提高可读性
            try:
                bg_pad = 2
                bg_x1 = max(0, label_x - bg_pad)
                bg_y1 = max(0, label_y - bg_pad)
                bg_x2 = min(w, label_x + text_w + bg_pad)
                bg_y2 = min(h, label_y + text_h + bg_pad)
                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 180))
            except Exception:
                pass
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

def calculate_house_center_of_mass(rooms_data):
    """计算房屋的重心（质心），基于房间面积加权
    
    Args:
        rooms_data: 房间数据列表
        
    Returns:
        (center_x, center_y): 重心坐标
    """
    if not rooms_data:
        return None, None
    
    total_weighted_x = 0
    total_weighted_y = 0
    total_area = 0
    
    for room in rooms_data:
        bbox = room.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2") 
        y2 = bbox.get("y2")
        
        if all(v is not None for v in [x1, y1, x2, y2]):
            # 计算房间中心和面积
            room_center_x = (x1 + x2) / 2
            room_center_y = (y1 + y2) / 2
            room_area = (x2 - x1) * (y2 - y1)
            
            # 根据房间类型设置权重
            room_type = room.get("type", "")
            if room_type in ["客厅", "餐厅"]:
                weight = 1.5  # 客厅餐厅权重更高
            elif room_type in ["卧室", "主卧", "次卧"]:
                weight = 1.2  # 卧室权重稍高
            elif room_type in ["厨房", "卫生间", "洗手间"]:
                weight = 0.8  # 厨卫权重稍低
            elif room_type in ["阳台", "储藏室", "衣帽间"]:
                weight = 0.5  # 辅助空间权重更低
            else:
                weight = 1.0  # 默认权重
            
            weighted_area = room_area * weight
            total_weighted_x += room_center_x * weighted_area
            total_weighted_y += room_center_y * weighted_area
            total_area += weighted_area
    
    if total_area > 0:
        center_x = total_weighted_x / total_area
        center_y = total_weighted_y / total_area
        return int(center_x), int(center_y)
    else:
        return None, None


def draw_twentyfour_mountains(image, polygon=None, north_angle=0, overlay_alpha=0.7, rooms_data=None):
    """绘制二十四山系统图 - 双层设计：内层八卦，外层二十四山"""
    # 转换为PIL图像以支持中文绘制
    pil_image = cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    
    h, w = image.shape[:2]
    
    # 计算太极点位置（与八宅八星图使用相同逻辑）
    if polygon:
        # 使用多边形数据计算最小外接圆的中心作为太极点
        center_x, center_y, circle_radius = get_minimum_enclosing_circle(polygon)
        center_x, center_y = int(center_x), int(center_y)
        print(f"二十四山太极点（多边形最小外接圆）: 中心({center_x}, {center_y}), 半径{circle_radius:.1f}")
    elif rooms_data:
        # 使用房间数据计算最小外接圆的中心
        center_x, center_y, circle_radius = get_minimum_enclosing_circle_from_rooms(rooms_data, w, h)
        center_x, center_y = int(center_x), int(center_y)
        print(f"二十四山太极点（房间数据最小外接圆）: 中心({center_x}, {center_y}), 半径{circle_radius:.1f}")
    else:
        # 使用图像中心
        center_x = w // 2
        center_y = h // 2
        circle_radius = min(w, h) / 2 - 10
        print(f"二十四山太极点（图像中心）: 中心({center_x}, {center_y})")

    # 计算合适的半径，外层圆完整包裹户型图
    base_radius = circle_radius
    base_radius = min(base_radius, min(w, h) / 2 - 10)

    # 设置三层圆环半径
    outer_radius = int(base_radius)          # 外层：二十四山
    middle_radius = int(base_radius * 0.7)   # 中层：八卦
    inner_radius = int(base_radius * 0.25)   # 内层：中心（缩小一半）
    
    # 创建透明覆盖层
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 获取字体（调大字体）
    mountain_font = get_chinese_font(30)     
    bagua_font = get_chinese_font(34)       
    title_font = get_chinese_font(36)       
    
    # 绘制整个二十四山圆形的半透明白色背景
    overlay_draw.ellipse([center_x - outer_radius, center_y - outer_radius,
                         center_x + outer_radius, center_y + outer_radius],
                        fill=(255, 255, 255, 100), outline=None)
    
    # 绘制圆环边界（黑色实线）
    overlay_draw.ellipse([center_x - outer_radius, center_y - outer_radius,
                         center_x + outer_radius, center_y + outer_radius],
                        outline=(0, 0, 0, 255), width=2)
    overlay_draw.ellipse([center_x - middle_radius, center_y - middle_radius,
                         center_x + middle_radius, center_y + middle_radius],
                        outline=(0, 0, 0, 255), width=2)
    overlay_draw.ellipse([center_x - inner_radius, center_y - inner_radius,
                         center_x + inner_radius, center_y + inner_radius],
                        outline=(0, 0, 0, 255), width=2)
    
    # 八卦数据定义
    bagua_data = [
        {"name": "坎", "angle": 0.0, "color": (100, 149, 237)},      # 正北
        {"name": "艮", "angle": 45.0, "color": (160, 82, 45)},      # 东北
        {"name": "震", "angle": 90.0, "color": (34, 139, 34)},      # 正东
        {"name": "巽", "angle": 135.0, "color": (0, 191, 255)},     # 东南
        {"name": "离", "angle": 180.0, "color": (220, 20, 60)},     # 正南
        {"name": "坤", "angle": 225.0, "color": (255, 165, 0)},     # 西南
        {"name": "兑", "angle": 270.0, "color": (169, 169, 169)},   # 正西
        {"name": "乾", "angle": 315.0, "color": (75, 0, 130)},      # 西北
    ]
    
    # 绘制八卦扇形边界（每个宫位45°的扇形）
    for i, bagua in enumerate(bagua_data):
        # 计算当前宫位的起始和结束角度
        start_angle = (bagua["angle"] - 22.5 + north_angle) % 360  # 每个宫位占45°，从中心向两边各22.5°
        end_angle = (bagua["angle"] + 22.5 + north_angle) % 360
        
        # 绘制宫位扇形的两条边界线
        for angle in [start_angle, end_angle]:
            radian = math.radians(angle)
            line_inner_x = center_x + inner_radius * math.sin(radian)
            line_inner_y = center_y - inner_radius * math.cos(radian)
            line_middle_x = center_x + middle_radius * math.sin(radian)
            line_middle_y = center_y - middle_radius * math.cos(radian)
            
            overlay_draw.line([line_inner_x, line_inner_y, line_middle_x, line_middle_y],
                             fill=(0, 0, 0, 255), width=2)
        
        # 计算八卦名称位置（位于扇形中央）
        # 八卦的实际角度（扇形中心）
        actual_bagua_angle = (bagua["angle"] + north_angle) % 360
        bagua_radian = math.radians(actual_bagua_angle)
        
        bagua_text_radius = (middle_radius + inner_radius) / 2
        bagua_text_x = center_x + bagua_text_radius * math.sin(bagua_radian)
        bagua_text_y = center_y - bagua_text_radius * math.cos(bagua_radian)
        
        # 绘制八卦名称
        if bagua_font:
            bbox = overlay_draw.textbbox((0, 0), bagua["name"], font=bagua_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 绘制半透明白色背景
            bg_padding = 4
            overlay_draw.rectangle([bagua_text_x - text_w//2 - bg_padding, 
                                   bagua_text_y - text_h//2 - bg_padding,
                                   bagua_text_x + text_w//2 + bg_padding, 
                                   bagua_text_y + text_h//2 + bg_padding],
                                  fill=(255, 255, 255, 180))
            
            # 绘制八卦名称（红色，突出显示）
            overlay_draw.text((bagua_text_x - text_w//2, bagua_text_y - text_h//2), 
                            bagua["name"], font=bagua_font, fill=(255, 0, 0, 255))
    
    # 绘制二十四山扇形边界（每个山位15°的扇形）
    for i in range(24):
        # 每个山位15°，计算扇形的边界角度
        mountain_center_angle = i * 15  # 山位中心角度
        start_angle = (mountain_center_angle - 7.5 + north_angle) % 360  # 扇形起始角度
        end_angle = (mountain_center_angle + 7.5 + north_angle) % 360    # 扇形结束角度
        
        # 绘制山位扇形的两条边界线
        for angle in [start_angle, end_angle]:
            radian = math.radians(angle)
            line_middle_x = center_x + middle_radius * math.sin(radian)
            line_middle_y = center_y - middle_radius * math.cos(radian)
            line_outer_x = center_x + outer_radius * math.sin(radian)
            line_outer_y = center_y - outer_radius * math.cos(radian)
            
            overlay_draw.line([line_middle_x, line_middle_y, line_outer_x, line_outer_y],
                             fill=(0, 0, 0, 255), width=1)
    
    # 绘制二十四山名称（在每个山位的中心）
    # 使用标准的二十四山顺序，子位在0°正北，然后整体旋转
    # 标准二十四山顺序（从子开始顺时针）
    standard_mountain_order = [
        "子", "癸", "丑", "艮", "寅", "甲", "卯", "乙", "辰", "巽", "巳", "丙",
        "午", "丁", "未", "坤", "申", "庚", "酉", "辛", "戌", "乾", "亥", "壬"
    ]
    
    # 创建山位名称到数据的映射
    mountain_map = {m["name"]: m for m in TWENTY_FOUR_MOUNTAINS}
    
    for i, mountain_name in enumerate(standard_mountain_order):
        if mountain_name in mountain_map:
            mountain = mountain_map[mountain_name]
            
            # 计算标准位置：子位在0°，每个山位占15°
            # 山位文字应该位于扇形的中央
            # 第i个山位的中心角度应该是 i * 15°（不需要+7.5，因为子位就在0°中心）
            standard_angle = i * 15  # 扇形中心角度
            
            # 应用北角偏移进行整体旋转
            actual_angle = (standard_angle + north_angle) % 360
            radian = math.radians(actual_angle)
            
            # 计算山名位置（在外层圆环中央）
            mountain_text_radius = (outer_radius + middle_radius) / 2
            mountain_text_x = center_x + mountain_text_radius * math.sin(radian)
            mountain_text_y = center_y - mountain_text_radius * math.cos(radian)
            
            # 绘制山名
            if mountain_font:
                # 计算文字大小用于居中
                bbox = overlay_draw.textbbox((0, 0), mountain["name"], font=mountain_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 根据山的类型设置颜色
                if mountain["type"] == "八卦":
                    text_color = (255, 0, 0, 255)      # 八卦用红色
                elif mountain["type"] == "天干":
                    text_color = (0, 0, 255, 255)      # 天干用蓝色
                else:  # 地支
                    text_color = (0, 128, 0, 255)      # 地支用绿色
                
                # 绘制半透明白色背景
                bg_padding = 3
                overlay_draw.rectangle([mountain_text_x - text_w//2 - bg_padding, 
                                       mountain_text_y - text_h//2 - bg_padding,
                                       mountain_text_x + text_w//2 + bg_padding, 
                                       mountain_text_y + text_h//2 + bg_padding],
                                      fill=(255, 255, 255, 180))
                
                # 绘制文字
                overlay_draw.text((mountain_text_x - text_w//2, mountain_text_y - text_h//2), 
                                mountain["name"], font=mountain_font, fill=text_color)
    
    # 绘制中心区域（缩小的中宫，不显示文字）
    center_radius = inner_radius * 0.6  # 进一步缩小中心区域
    overlay_draw.ellipse([center_x - center_radius, center_y - center_radius,
                         center_x + center_radius, center_y + center_radius],
                        fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))
    
    # 在圆外面添加八个方位标识
    direction_font = get_chinese_font(48)  # 方位标识字体
    direction_labels = [
        {"name": "北", "angle": 0.0},
        {"name": "东北", "angle": 45.0},
        {"name": "东", "angle": 90.0},
        {"name": "东南", "angle": 135.0},
        {"name": "南", "angle": 180.0},
        {"name": "西南", "angle": 225.0},
        {"name": "西", "angle": 270.0},
        {"name": "西北", "angle": 315.0}
    ]
    
    # 方位标识距离圆心的半径（在外圆之外）
    direction_radius = outer_radius + 40
    
    for direction in direction_labels:
        # 计算实际角度（考虑north_angle旋转）
        actual_angle = (direction["angle"] + north_angle) % 360
        radian = math.radians(actual_angle)
        
        # 计算方位标识位置
        dir_x = center_x + direction_radius * math.sin(radian)
        dir_y = center_y - direction_radius * math.cos(radian)
        
        if direction_font:
            # 计算文字大小用于居中
            bbox = overlay_draw.textbbox((0, 0), direction["name"], font=direction_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 绘制方位标识（深蓝色，清晰可见）
            overlay_draw.text((dir_x - text_w//2, dir_y - text_h//2), 
                            direction["name"], font=direction_font, fill=(0, 0, 139, 255))
    
    # 将透明overlay合成到原图上
    pil_image = pil_image.convert('RGBA')
    result = Image.alpha_composite(pil_image, overlay)
    
    # 转换回OpenCV格式
    return pil_to_cv2(result.convert('RGB'))

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

def create_combined_visualization(
    image,
    rooms_data,
    direction_stars_mapping,
    polygon_luoshu=None,
    polygon_full=None,
    missing_corners=None,
    house_orientation=None,
    north_angle=0,
):
    """创建组合可视化图像：九宫格图 + 八宅八星圆形图 + 二十四山系统图，包含缺角信息，上中下布局"""
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
    adjusted_polygon_luoshu = polygon_luoshu
    adjusted_polygon_full = polygon_full
    adjusted_rooms = rooms_data  # 默认使用原始房间数据

    if polygon_luoshu:
        # 调整九宫格用多边形坐标
        adjusted_polygon_luoshu = [(x + x_offset, y + y_offset) for x, y in polygon_luoshu]
    if polygon_full:
        # 调整八宅与二十四山用多边形坐标
        adjusted_polygon_full = [(x + x_offset, y + y_offset) for x, y in polygon_full]
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

    # 在底图上先绘制房间标签，避免覆盖后续方位与星位文字
    base_with_rooms = draw_room_positions(extended_image.copy(), adjusted_rooms)
    
    # 创建两个分离的图像
    if missing_corners:
        luoshu_image = draw_luoshu_grid_with_missing_corners(
            base_with_rooms.copy(),
            adjusted_rooms,
            adjusted_polygon_luoshu,
            missing_corners=missing_corners,
            north_angle=north_angle,
        )
    else:
        luoshu_image = draw_luoshu_grid_only(
            base_with_rooms.copy(), adjusted_polygon_luoshu, north_angle=north_angle
        )

    bazhai_image = draw_bazhai_circle(
        base_with_rooms.copy(),
        direction_stars_mapping,
        adjusted_polygon_full,
        adjusted_rooms,
        house_orientation,
        north_angle=north_angle,
    )

    # 创建二十四山系统图
    mountains_image = draw_twentyfour_mountains(
        base_with_rooms.copy(),
        adjusted_polygon_full,
        north_angle=north_angle,
        rooms_data=rooms_data,
    )
    
    # 房间位置已先行绘制在底图上，避免遮挡星位与方位标签
    
    # 垂直拼接三张图（上中下布局）
    # combined_image = np.vstack([luoshu_image, bazhai_image, mountains_image])
    
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
        # 中间图像标题（八宅八星）
        middle_title = "八宅八星图"
        bbox = draw2.textbbox((0, 0), middle_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        center_x = extended_w // 2
        draw2.text((center_x - title_w//2, 15), middle_title, font=title_font, fill=(255, 255, 255))
    
    title_cv2_2 = pil_to_cv2(pil_title2)
    bazhai_with_title = np.vstack([title_cv2_2, bazhai_image])
    
    # 为二十四山系统图添加标题
    title_area3 = np.zeros((title_height, extended_w, 3), dtype=np.uint8)
    title_area3[:] = (50, 50, 50)
    
    pil_title3 = cv2_to_pil(title_area3)
    draw3 = ImageDraw.Draw(pil_title3)
    
    if title_font:
        # 下方图像标题（二十四山系统）
        bottom_title = "二十四山系统图"
        bbox = draw3.textbbox((0, 0), bottom_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        center_x = extended_w // 2
        draw3.text((center_x - title_w//2, 15), bottom_title, font=title_font, fill=(255, 255, 255))
    
    title_cv2_3 = pil_to_cv2(pil_title3)
    mountains_with_title = np.vstack([title_cv2_3, mountains_image])
    
    # 最终垂直拼接三张图
    final_image = np.vstack([luoshu_with_title, bazhai_with_title, mountains_with_title])
    
    return final_image


def create_combined_visualization_old(image, rooms_data, direction_stars_mapping, polygon=None, north_angle=0):
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

        # 同时读取原始JSON数据
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        rooms = raw_data.get("rooms", [])
        # 分别生成供不同分析使用的多边形
        polygon_full = create_polygon_from_rooms(rooms, shrink_balcony=False)
        polygon_luoshu = create_polygon_from_rooms(rooms, shrink_balcony=True)

    except Exception as e:
        print(f"加载户型图数据失败: {e}")
        raise
    
    # 获取图像路径和北向角度
    meta = raw_data.get('meta', {})
    original_image_path = meta.get('original_image')
    result_image_path = meta.get('result_image') or meta.get('output_image')
    north_angle = meta.get('north_angle', 0)  # 默认0度（上方为北）
    
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
    star_analysis = analyze_eightstars(polygon_full, room_data, doc, gua)
    
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
    
    # 执行缺角分析 - 使用阳台缩减一半的房间数据
    rooms_for_analysis = []
    for room in rooms:
        bbox = room.get("bbox", {})
        room_type = str(room.get("type", ""))
        if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
            # 复制bbox数据
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            
            # 如果是阳台，按照九宫格规则缩减面积
            if room_type == "阳台":
                w = x2 - x1  # 宽度
                h = y2 - y1  # 高度
                if abs(w) <= abs(h):  # 竖向阳台，缩减宽度
                    x2 = (x1 + x2) / 2.0
                else:  # 横向阳台，缩减高度
                    y1 = (y1 + y2) / 2.0
            rooms_for_analysis.append({"bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
    
    # 添加详细的面积计算调试信息
    print(f"\n=== 详细面积计算调试信息 ===")
    print(f"图像尺寸: {doc.img_w} x {doc.img_h}")
    print(f"北向角度: {north_angle}度")
    
    # 计算房屋外接矩形
    if rooms_for_analysis:
        boxes = []
        for room in rooms_for_analysis:
            bbox = room["bbox"]
            boxes.append((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
        
        min_x = min(b[0] for b in boxes)
        min_y = min(b[1] for b in boxes)
        max_x = max(b[2] for b in boxes)
        max_y = max(b[3] for b in boxes)
        
        print(f"房屋外接矩形: ({min_x:.1f}, {min_y:.1f}) 到 ({max_x:.1f}, {max_y:.1f})")
        print(f"房屋总宽度: {max_x - min_x:.1f}, 总高度: {max_y - min_y:.1f}")
        
        # 九宫格划分
        grid_w = (max_x - min_x) / 3.0
        grid_h = (max_y - min_y) / 3.0
        print(f"九宫格单元大小: {grid_w:.1f} x {grid_h:.1f}")
        
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        print(f"房屋中心: ({center_x:.1f}, {center_y:.1f})")
        
        # 计算各宫格的详细信息
        direction_names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        
        print(f"\n=== 各宫格面积计算详情 ===")
        for gy in range(3):
            for gx in range(3):
                # 跳过中心区域
                if gx == 1 and gy == 1:
                    continue
                    
                # 计算九宫格区域
                region_x1 = min_x + gx * grid_w
                region_x2 = min_x + (gx + 1) * grid_w
                region_y1 = min_y + gy * grid_h
                region_y2 = min_y + (gy + 1) * grid_h
                
                # 计算该区域被房间覆盖的面积
                region_area = (region_x2 - region_x1) * (region_y2 - region_y1)
                covered_area = 0
                
                print(f"\n宫格[{gx},{gy}] 区域: ({region_x1:.1f}, {region_y1:.1f}) 到 ({region_x2:.1f}, {region_y2:.1f})")
                print(f"宫格总面积: {region_area:.1f}")
                
                # 计算方位
                region_cx = (region_x1 + region_x2) / 2.0
                region_cy = (region_y1 + region_y2) / 2.0
                dx = region_cx - center_x
                dy = region_cy - center_y
                angle = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
                angle = (angle - north_angle + 360.0) % 360.0
                idx = int(((angle + 22.5) % 360) / 45)
                direction = direction_names[idx]
                
                print(f"宫格中心: ({region_cx:.1f}, {region_cy:.1f})")
                print(f"相对位移: dx={dx:.1f}, dy={dy:.1f}")
                print(f"计算角度: {angle:.1f}度, 方位: {direction}")
                
                # 检查每个房间与此宫格的重叠
                print(f"房间重叠情况:")
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    # 计算房间与九宫格区域的重叠
                    overlap_x1 = max(x1, region_x1)
                    overlap_y1 = max(y1, region_y1)
                    overlap_x2 = min(x2, region_x2)
                    overlap_y2 = min(y2, region_y2)
                    
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        covered_area += overlap_area
                        print(f"  房间{i+1}: ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) 重叠面积: {overlap_area:.1f}")
                    else:
                        print(f"  房间{i+1}: ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) 无重叠")
                
                # 计算覆盖率
                coverage_ratio = covered_area / region_area if region_area > 0 else 0
                missing_ratio = 1.0 - coverage_ratio
                
                print(f"总覆盖面积: {covered_area:.1f}")
                print(f"覆盖率: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")
                print(f"缺角率: {missing_ratio:.3f} ({missing_ratio*100:.1f}%)")
                
                if direction == "坤":
                    print(f"*** 这是坤宫 - 右上角宫格 ***")

    missing_corners = analyze_missing_corners_by_room_coverage(
        rooms_for_analysis, doc.img_w, doc.img_h, threshold=0.75, north_angle=north_angle
    )
    
    if missing_corners:
        print(f"\n=== 缺角分析结果 ===")
        for corner in missing_corners:
            missing_rate = 1.0 - corner['coverage']
            print(f"    {corner['direction']}方: 覆盖率 {corner['coverage']:.3f} ({corner['coverage']*100:.1f}%), 缺角率 {missing_rate:.3f} ({missing_rate*100:.1f}%)")
    else:
        print(f"\n=== 缺角分析结果 === 无明显缺角")
    
    # 创建组合可视化图像（包含缺角信息）
    house_orientation = getattr(doc, 'house_orientation', '坐北朝南')
    final_image = create_combined_visualization(
        image,
        rooms,
        direction_stars_mapping,
        polygon_luoshu,
        polygon_full,
        missing_corners,
        house_orientation,
        north_angle,
    )
    
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
    parser.add_argument('json_path', nargs='?', default='.\\output\\demo15_result_edited.json', help='输入的JSON文件路径 (默认: .\\output\\demo15_result_edited.json)')
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
