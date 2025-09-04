"""Identify missing corners of a floorplan using a 3x3 Luo Shu grid."""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Dict

import numpy as np
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

# Default orientation parameters; callers may override these module level variables
NORTH_ANGLE: int = 0  # 0=North, 90=East as used in editor.models
HOUSE_ORIENTATION: str = "坐北朝南"

DIRECTION_NAMES = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]

# Mapping from direction to simple fengshui suggestions
SUGGESTIONS: Dict[str, str] = {
    "东": "东侧缺角可摆放绿植以增旺木气。",
    "东北": "东北缺角建议保持光线充足，放置瓷器装饰。",
    "北": "北方缺角可用水景或蓝色饰品来化解。",
    "西北": "西北缺角可悬挂金属风铃增强金气。",
    "西": "西侧缺角宜放置白色或金属饰品以补足。",
    "西南": "西南缺角可摆放陶土饰物或黄色家具。",
    "南": "南方缺角可采用红色灯光或木饰品改善。",
    "东南": "东南缺角可种植常绿植物以提升生气。",
}

# Generic remedies applicable to任何缺角的常见化解思路
GENERAL_REMEDIES: List[str] = [
    "使用功能性家具\u201c补角\u201d，如书柜、衣柜贴墙摆放。",
    "在缺角方位摆放生机盎然的植物或装饰品，营造圆满感。",
    "利用镜面或柔和灯光扩大空间感，增强该方位气场。",
    "结合房屋用途在相邻方位调整布局，以弥补不利影响。",
]


def general_remedies() -> List[str]:
    """Return common remedies for Luo Shu missing-corner issues."""
    return GENERAL_REMEDIES.copy()


def _direction_from_point(cx: int, cy: int, img_w: int, img_h: int, north_angle: int) -> str:
    """Convert a point to a compass direction considering north angle."""
    dx = cx - img_w / 2.0
    dy = cy - img_h / 2.0
    
    # 处理中心位置
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "中"
    
    # 计算角度（PIL坐标系：右为0度，下为90度）
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    
    # 根据north_angle调整角度
    # 新系统：north_angle直接表示北方角度，0°=北
    angle = (angle - north_angle + 360.0) % 360.0
    
    # 转换为方向索引
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]


def analyze_missing_corners_by_room_coverage(
    rooms: List[Dict],
    width: int,
    height: int,
    threshold: float = 0.75,
    north_angle: int = 0,
) -> List[Dict[str, object]]:
    """基于房间覆盖率分析缺角（修正版本，避免凸包算法的过度连接问题）
    
    Parameters
    ----------
    rooms: list of dict
        房间数据，每个dict包含bbox信息
    width, height: int
        图像尺寸
    threshold: float, optional
        最小覆盖率要求，低于此值认为缺角
        
    Returns
    -------
    list of dict
        每个dict包含`direction`, `coverage`, 和`suggestion`键
    """
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
    
    # 计算房屋的外接矩形
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    
    # 九宫格划分
    grid_w = (max_x - min_x) / 3.0
    grid_h = (max_y - min_y) / 3.0
    
    missing = []
    
    # 九宫格方位映射（根据north_angle动态调整）
    # (gx, gy) -> 方位名称
    if north_angle == 0:  # 上方是北方（新的默认）
        grid_directions = {
            (0, 0): "西北", (1, 0): "北", (2, 0): "东北",
            (0, 1): "西",   (1, 1): "中", (2, 1): "东",
            (0, 2): "西南", (1, 2): "南", (2, 2): "东南"
        }
    elif north_angle == 90:  # 上方是东方
        grid_directions = {
            (0, 0): "东北", (1, 0): "东", (2, 0): "东南",
            (0, 1): "北",   (1, 1): "中", (2, 1): "南",
            (0, 2): "西北", (1, 2): "西", (2, 2): "西南"
        }
    elif north_angle == 180:  # 上方是南方
        grid_directions = {
            (0, 0): "东南", (1, 0): "南", (2, 0): "西南",
            (0, 1): "东",   (1, 1): "中", (2, 1): "西",
            (0, 2): "东北", (1, 2): "北", (2, 2): "西北"
        }
    elif north_angle == 270:  # 上方是西方
        grid_directions = {
            (0, 0): "西南", (1, 0): "西", (2, 0): "西北",
            (0, 1): "南",   (1, 1): "中", (2, 1): "北",
            (0, 2): "东南", (1, 2): "东", (2, 2): "东北"
        }
    else:
        # 其他角度使用默认映射
        grid_directions = {
            (0, 0): "西北", (1, 0): "北", (2, 0): "东北",
            (0, 1): "西",   (1, 1): "中", (2, 1): "东",
            (0, 2): "西南", (1, 2): "南", (2, 2): "东南"
        }
    
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
            
            for x1, y1, x2, y2 in boxes:
                # 计算房间与九宫格区域的重叠
                overlap_x1 = max(x1, region_x1)
                overlap_y1 = max(y1, region_y1)
                overlap_x2 = min(x2, region_x2)
                overlap_y2 = min(y2, region_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    covered_area += overlap_area
            
            # 计算覆盖率
            coverage_ratio = covered_area / region_area if region_area > 0 else 0
            
            # 判断是否缺角
            if coverage_ratio < threshold:
                # 使用网格位置直接确定方位
                direction = grid_directions.get((gx, gy), "未知")
                suggestion = SUGGESTIONS.get(direction, "可通过合理布置改善气场。")
                
                missing.append({
                    "direction": direction,
                    "coverage": round(coverage_ratio, 3),
                    "suggestion": suggestion,
                })
    
    return missing


def analyze_missing_corners(
    polygon_points: Iterable[Tuple[float, float]],
    width: int,
    height: int,
    threshold: float = 0.75,
) -> List[Dict[str, object]]:
    """Analyze which sectors of a 3x3 grid are missing from the floorplan.

    Parameters
    ----------
    polygon_points: iterable of (x, y)
        Vertices of the floorplan polygon.
    width, height: int
        Size of the image containing the polygon.
    threshold: float, optional
        Minimum coverage ratio required to consider the sector present.

    Returns
    -------
    list of dict
        Each dict contains `direction`, `coverage`, and `suggestion` keys.
    """
    pts = np.array(list(polygon_points), dtype=np.float32)
    if pts.size == 0:
        return []

    # Compute overall bounds
    min_x = float(np.min(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_x = float(np.max(pts[:, 0]))
    max_y = float(np.max(pts[:, 1]))

    # Helper: coverage ratio within an axis-aligned cell
    def _coverage_ratio(x1: int, y1: int, x2: int, y2: int) -> float:
        if x2 <= x1 or y2 <= y1:
            return 0.0
        if _HAS_CV2:
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)  # type: ignore[arg-type]
            cell = mask[y1:y2, x1:x2]
            total = cell.size
            if total <= 0:
                return 0.0
            return float(cell.sum()) / float(total)
        # Fallback: sample grid + point-in-polygon via ray casting
        sx = max(1, min(16, x2 - x1))
        sy = max(1, min(16, y2 - y1))
        xs = np.linspace(x1 + 0.5, x2 - 0.5, sx)
        ys = np.linspace(y1 + 0.5, y2 - 0.5, sy)
        gx, gy = np.meshgrid(xs, ys)
        P = np.stack([gx.ravel(), gy.ravel()], axis=1)
        poly = pts
        x = P[:, 0][:, None]
        y = P[:, 1][:, None]
        x1p = poly[:, 0][None, :]
        y1p = poly[:, 1][None, :]
        x2p = np.roll(poly[:, 0], -1)[None, :]
        y2p = np.roll(poly[:, 1], -1)[None, :]
        cond = ((y1p > y) != (y2p > y)) & (
            x < (x2p - x1p) * (y - y1p) / (y2p - y1p + 1e-9) + x1p
        )
        inside = (np.count_nonzero(cond, axis=1) % 2) == 1
        return float(np.count_nonzero(inside)) / float(P.shape[0])

    grid_w = (max_x - min_x) / 3.0
    grid_h = (max_y - min_y) / 3.0

    missing: List[Dict[str, object]] = []
    for gy in range(3):
        for gx in range(3):
            x1 = int(min_x + gx * grid_w)
            x2 = int(min_x + (gx + 1) * grid_w)
            y1 = int(min_y + gy * grid_h)
            y2 = int(min_y + (gy + 1) * grid_h)
            if x2 <= x1 or y2 <= y1:
                continue
            ratio = _coverage_ratio(x1, y1, x2, y2)
            if ratio < threshold:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                direction = _direction_from_point(cx, cy, width, height, NORTH_ANGLE)
                suggestion = SUGGESTIONS.get(direction, "可通过合理布置改善气场。")
                missing.append(
                    {
                        "direction": direction,
                        "coverage": round(ratio, 3),
                        "suggestion": suggestion,
                    }
                )
    return missing
