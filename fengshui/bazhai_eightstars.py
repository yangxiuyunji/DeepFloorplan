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
    "坎": {
        "东南": "生气",
        "南": "延年",
        "东": "天医",
        "北": "伏位",
        "西南": "绝命",
        "东北": "五鬼",
        "西北": "六煞",
        "西": "祸害"
    },
    "离": {
        "东南": "天医",
        "南": "伏位",
        "东": "生气",
        "北": "延年",
        "西南": "六煞",
        "东北": "祸害",
        "西北": "绝命",
        "西": "五鬼"
    },
    "震": {
        "东南": "延年",
        "南": "生气",
        "东": "伏位",
        "北": "天医",
        "西南": "祸害",
        "东北": "六煞",
        "西北": "五鬼",
        "西": "绝命"
    },
    "巽": {
        "东南": "伏位",
        "南": "天医",
        "东": "延年",
        "北": "生气",
        "西南": "五鬼",
        "东北": "绝命",
        "西北": "祸害",
        "西": "六煞"
    },
    "乾": {
        "东南": "祸害",
        "南": "绝命",
        "东": "五鬼",
        "北": "六煞",
        "西南": "延年",
        "东北": "天医",
        "西北": "伏位",
        "西": "生气"
    },
    "兑": {
        "东南": "六煞",
        "南": "五鬼",
        "东": "绝命",
        "北": "祸害",
        "西南": "天医",
        "东北": "延年",
        "西北": "生气",
        "西": "伏位"
    },
    "艮": {
        "东南": "绝命",
        "南": "祸害",
        "东": "六煞",
        "北": "五鬼",
        "西南": "生气",
        "东北": "伏位",
        "西北": "天医",
        "西": "延年"
    },
    "坤": {
        "东南": "五鬼",
        "南": "六煞",
        "东": "祸害",
        "北": "绝命",
        "西南": "伏位",
        "东北": "生气",
        "西北": "延年",
        "西": "天医"
    }
}

# Mapping from house orientation (坐向) to the fixed eight-star distribution.
# Eight house types correspond to the eight trigrams; here we map the common
# Chinese orientation phrases to the same direction→star tables used for Ming
# Gua above. Users can therefore obtain star information even when no personal
# "命卦" is specified.
HOUSE_DIRECTION_STARS: Dict[str, Dict[str, str]] = {
    "坐北朝南": GUA_DIRECTION_STARS["坎"],
    "坐南朝北": GUA_DIRECTION_STARS["离"],
    "坐东朝西": GUA_DIRECTION_STARS["震"],
    "坐西朝东": GUA_DIRECTION_STARS["兑"],
    "坐东南朝西北": GUA_DIRECTION_STARS["巽"],
    "坐东北朝西南": GUA_DIRECTION_STARS["艮"],
    "坐西北朝东南": GUA_DIRECTION_STARS["乾"],
    "坐西南朝东北": GUA_DIRECTION_STARS["坤"],
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

# General guidelines for applying BaZhai adjustments
GENERAL_GUIDELINES: List[str] = [
    "根据出生年份确定居住者命卦，判定东四命或西四命。",
    "主要房间尽量安排在与命卦相合的吉方，如卧室、书房等。",
    "凶星方位宜作次要空间，吉星方布置核心功能区。",
    "依五行调配颜色与摆设，并保持良好通风采光。",
    "以实用与舒适为先，如无法变动位置可采取折中化解。",
]


def general_guidelines() -> List[str]:
    """Return generic layout guidelines for the BaZhai eight-star method."""
    return GENERAL_GUIDELINES.copy()


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

    # Determine the direction→star mapping either from personal Ming Gua or
    # from the house orientation. If neither is provided we fall back to an
    # empty mapping which yields generic suggestions only.
    if gua:
        direction_stars = GUA_DIRECTION_STARS.get(gua, {})
    else:
        direction_stars = HOUSE_DIRECTION_STARS.get(house_orientation, {})

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
        star = direction_stars.get(direction)
        nature = ""
        suggestion = "根据方位合理布置。"
        if star:
            nature, suggestion = STAR_INFO.get(star, ("", suggestion))
        name = (
            room.get("name")
            or room.get("type")
            or str(room.get("id", "room"))
        )
        item = {"room": name, "direction": direction}
        if star:
            item.update({"star": star, "nature": nature, "suggestion": suggestion})
        else:
            item.update({"star": "", "nature": "", "suggestion": suggestion})
        results.append(item)
    return results
