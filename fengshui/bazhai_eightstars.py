"""Analyze rooms' orientations using BaZhai (Eight Mansions) fengshui."""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Tuple, Dict, Any, List

# Default orientation parameters; callers may override these module level variables
NORTH_ANGLE: int = 90  # 0=East, 90=North as used in editor.models
HOUSE_ORIENTATION: str = "坐北朝南"

DIRECTION_NAMES = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]

# Mapping from Ming Gua to direction -> star
GUA_DIRECTION_STARS: Dict[str, Dict[str, str]] = {
    # East group
    "坎": {
        "东": "天医",
        "东南": "生气",
        "南": "延年",
        "北": "伏位",
        "西": "五鬼",
        "西北": "祸害",
        "西南": "绝命",
        "东北": "六煞",
    },
    "震": {
        "东": "伏位",
        "东南": "延年",
        "南": "生气",
        "北": "天医",
        "西": "绝命",
        "西北": "六煞",
        "西南": "祸害",
        "东北": "五鬼",
    },
    "巽": {
        "东": "生气",
        "东南": "伏位",
        "南": "延年",
        "北": "天医",
        "西": "六煞",
        "西北": "绝命",
        "西南": "五鬼",
        "东北": "祸害",
    },
    "离": {
        "东": "生气",
        "东南": "延年",
        "南": "伏位",
        "北": "天医",
        "西": "祸害",
        "西北": "绝命",
        "西南": "六煞",
        "东北": "五鬼",
    },
    # West group
    "坤": {
        "西": "天医",
        "西北": "延年",
        "西南": "伏位",
        "东北": "生气",
        "东": "五鬼",
        "东南": "六煞",
        "北": "祸害",
        "南": "绝命",
    },
    "乾": {
        "西": "延年",
        "西北": "伏位",
        "西南": "天医",
        "东北": "生气",
        "东": "绝命",
        "东南": "祸害",
        "南": "六煞",
        "北": "五鬼",
    },
    "兑": {
        "西": "伏位",
        "西北": "生气",
        "西南": "天医",
        "东北": "延年",
        "东": "六煞",
        "东南": "绝命",
        "南": "五鬼",
        "北": "祸害",
    },
    "艮": {
        "西": "生气",
        "西北": "天医",
        "西南": "延年",
        "东北": "伏位",
        "东": "祸害",
        "东南": "五鬼",
        "南": "绝命",
        "北": "六煞",
    },
}

# Information about each star: nature and suggestion
STAR_INFO: Dict[str, Tuple[str, str]] = {
    "生气": ("吉", "宜设卧室或书房，利于事业与健康。"),
    "延年": ("吉", "适合做卧室、客厅，促进人际与婚姻和谐。"),
    "天医": ("吉", "适合厨房或卫生间，利于健康与疗愈。"),
    "伏位": ("吉", "可作为玄关或静室，增强稳定与自我。"),
    "祸害": ("凶", "宜布置为储物等次要空间，摆放金属饰物化解。"),
    "绝命": ("凶", "避免重要活动，可用金属或水法化解。"),
    "六煞": ("凶", "可设卫生间或储藏，搭配绿植以缓和。"),
    "五鬼": ("凶", "不宜久留，可用火光或红色调化解。"),
}


def _direction_from_point(cx: float, cy: float, ox: float, oy: float, north_angle: float) -> str:
    """Convert a point to compass direction considering north angle."""
    dx = cx - ox
    dy = cy - oy
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0  # 0=East, 90=North
    angle = (angle - north_angle + 360.0) % 360.0
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]


def analyze_eightstars(
    polygon_points: Iterable[Tuple[float, float]],
    rooms: Iterable[Mapping[str, Any]],
    orientation: Mapping[str, Any] | Any,
    gua: str | None = None,
) -> List[Dict[str, str]]:
    """Analyze each room's direction and corresponding BaZhai star.

    Parameters
    ----------
    polygon_points: iterable of (x, y)
        Vertices of the floorplan polygon to determine the house center.
    rooms: iterable of mapping
        Each room mapping should contain either ``center`` or ``bbox`` and a
        ``name``/``type`` field for display.
    orientation: mapping or object
        Should provide ``north_angle`` (0°=东, 90°=北) and ``house_orientation``.
    gua: str, optional
        Personal "命卦". If provided, directions will map to the eight stars.

    Returns
    -------
    list of dict
        Each dict contains ``room``, ``direction``, ``star`` and ``suggestion``.
    """
    pts = list(polygon_points)
    if not pts:
        return []
    xs, ys = zip(*pts)
    ox = sum(xs) / len(xs)
    oy = sum(ys) / len(ys)

    north_angle = getattr(orientation, "north_angle", None)
    if north_angle is None and isinstance(orientation, Mapping):
        north_angle = orientation.get("north_angle", NORTH_ANGLE)
    house_orientation = getattr(orientation, "house_orientation", None)
    if house_orientation is None and isinstance(orientation, Mapping):
        house_orientation = orientation.get("house_orientation", HOUSE_ORIENTATION)

    results: List[Dict[str, str]] = []
    for room in rooms:
        cx: float
        cy: float
        if "center" in room:
            cx, cy = room["center"]
        elif "bbox" in room:
            x1, y1, x2, y2 = room["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
        else:
            continue
        direction = _direction_from_point(cx, cy, ox, oy, north_angle)
        star = None
        nature = None
        suggestion = "根据方位合理布置。"
        if gua:
            star = GUA_DIRECTION_STARS.get(gua, {}).get(direction)
            if star:
                nature, suggestion = STAR_INFO.get(star, ("", suggestion))
        name = (
            room.get("name")
            or room.get("type")
            or str(room.get("id", "room"))
        )
        item = {"room": name, "direction": direction}
        if star:
            item.update({"star": star, "nature": nature or "", "suggestion": suggestion})
        else:
            item.update({"star": "", "nature": "", "suggestion": suggestion})
        results.append(item)
    return results
