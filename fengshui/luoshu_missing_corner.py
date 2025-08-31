"""Identify missing corners of a floorplan using a 3x3 Luo Shu grid."""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Dict

import cv2
import numpy as np

# Default orientation parameters; callers may override these module level variables
NORTH_ANGLE: int = 90  # 0=East, 90=North as used in editor.models
HOUSE_ORIENTATION: str = "坐北朝南"

DIRECTION_NAMES = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]

# Mapping from direction to simple fengshui suggestions
# Simple one-line suggestions retained for backward compatibility
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

# Five-element associations and example items for each direction
ELEMENT_ITEMS: Dict[str, Tuple[str, List[str]]] = {
    "东": ("木", ["绿植", "木制品"]),
    "东南": ("木", ["常绿植物", "木饰品"]),
    "南": ("火", ["红色灯光", "木质摆件"]),
    "西南": ("土", ["陶土饰物", "黄色家具"]),
    "西": ("金", ["白色饰品", "金属摆设"]),
    "西北": ("金", ["金属风铃", "白色金属饰品"]),
    "北": ("水", ["水景", "蓝色饰品"]),
    "东北": ("土", ["瓷器装饰", "土色摆件"]),
}


def rectification_steps(direction: str) -> List[str]:
    """Generate a step-by-step plan to remedy a missing corner.

    Parameters
    ----------
    direction: str
        The compass direction of the missing corner.

    Returns
    -------
    list of str
        Four textual steps guiding the user how to rectify the issue.
    """
    element, items = ELEMENT_ITEMS.get(direction, ("", []))
    item_str = "、".join(items) if items else "与方位相符的物件"
    return [
        f"确认缺角方位：{direction}方。",
        f"挑选与该方位相符的五行物件或植物（{element}），例如：{item_str}。",
        "在缺角位置布置镜子、屏风或植物，形成视觉补角。",
        "定期维护，确保摆放物品整洁、不枯败。",
    ]


def _direction_from_point(cx: int, cy: int, img_w: int, img_h: int, north_angle: int) -> str:
    """Convert a point to a compass direction considering north angle."""
    dx = cx - img_w / 2.0
    dy = cy - img_h / 2.0
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0  # 0=East, 90=North
    angle = (angle - north_angle + 360.0) % 360.0
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]


def analyze_missing_corners(
    polygon_points: Iterable[Tuple[float, float]],
    width: int,
    height: int,
    threshold: float = 0.6,
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
    pts = np.array(list(polygon_points), dtype=np.int32)
    if pts.size == 0:
        return []

    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

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
            cell = mask[y1:y2, x1:x2]
            total = cell.size
            cover = float(cell.sum())
            ratio = cover / total if total else 0.0
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
                        "steps": rectification_steps(direction),
                    }
                )
    return missing
