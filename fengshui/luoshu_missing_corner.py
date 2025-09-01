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
NORTH_ANGLE: int = 90  # 0=East, 90=North as used in editor.models
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
