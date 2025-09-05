from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

DIRECTION_NAMES = ["东","东北","北","西北","西","西南","南","东南"]

def compute_direction_8(cx: int, cy: int, img_w: int, img_h: int) -> str:
    """按生成脚本约定：0°=东，逆时针。"""
    dx = cx - img_w / 2.0
    dy = cy - img_h / 2.0
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0  # 0=东 90=北
    idx = int(((angle + 22.5) % 360) / 45)
    return DIRECTION_NAMES[idx]

@dataclass
class RoomModel:
    id: str
    type: str
    label_id: int
    bbox: tuple  # (x1,y1,x2,y2)
    text_raw: str = ""
    confidence: float = 0.0
    source: str = "ocr"
    edited: bool = False
    index: int = 1
    center: tuple = field(default_factory=lambda: (0, 0))
    center_normalized: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    area_pixels: int = 0
    distance_to_center: float = 0.0
    direction_8: str = "东"

    def recompute(self, img_w: int, img_h: int):
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.center = (cx, cy)
        self.center_normalized = {"x": round(cx / img_w, 4), "y": round(cy / img_h, 4)}
        self.area_pixels = int((x2 - x1 + 1) * (y2 - y1 + 1))
        dx = cx - img_w / 2.0
        dy = cy - img_h / 2.0
        self.distance_to_center = round((dx * dx + dy * dy) ** 0.5, 2)
        self.direction_8 = compute_direction_8(cx, cy, img_w, img_h)

    def update_bbox(self, bbox: tuple[int, int, int, int], img_w: int, img_h: int):
        x1, y1, x2, y2 = bbox
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # clamp
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 - x1 < 4 or y2 - y1 < 4:
            # 保持最小尺寸 5x5
            x2 = x1 + 4
            y2 = y1 + 4
            if x2 >= img_w:
                x1 = img_w - 5
                x2 = img_w - 1
            if y2 >= img_h:
                y1 = img_h - 5
                y2 = img_h - 1
        self.bbox = (x1, y1, x2, y2)
        self.edited = True
        self.recompute(img_w, img_h)

    def update_field(self, field_name: str, value, img_w: int, img_h: int):
        if field_name == "type":
            self.type = value
        elif field_name == "text_raw":
            self.text_raw = value
        elif field_name == "label_id":
            self.label_id = int(value)
        elif field_name == "confidence":
            self.confidence = float(value)
        self.edited = True
        self.recompute(img_w, img_h)

class FloorplanDocument:
    def __init__(self, json_path: 'Optional[str]' = None):
        self.json_path = json_path
        self.output_path = json_path
        # 当前编辑显示的背景图（可在原图与结果图之间切换）
        self.image_path = None  # 当前使用的底图路径
        # 结果图（原推理输出图）
        self.result_image_path = None
        # 原始输入图
        self.original_image_path = None
        # 可选 mask 图
        self.mask_path = None
        # 图像尺寸
        self.img_w = 0
        self.img_h = 0
        # 房间与类别
        self.rooms = []  # list[RoomModel]
        self.categories = []  # list[str]
        # label id 自增种子
        self._next_label_seed = 20
        # 记录类别 -> label_id 的映射（稳定复用）
        self.category_label_map = {}
        # 全局户型属性
        self.house_orientation = "坐北朝南"
        self.north_angle = 0  # 正北方角度，默认0度代表正北方
        self.magnetic_declination = 0  # 磁偏角，默认0度

    # ---------- 基础 ----------
    def set_image_meta(self, w: int, h: int):
        self.img_w, self.img_h = w, h

    def add_room(self, room: RoomModel):
        self.rooms.append(room)

    def get_room(self, rid: str) -> 'Optional[RoomModel]':
        for r in self.rooms:
            if r.id == rid:
                return r
        return None

    def ensure_indices(self):
        by_type: dict[str, list[RoomModel]] = {}
        for r in self.rooms:
            by_type.setdefault(r.type, []).append(r)
        for t, lst in by_type.items():
            lst.sort(key=lambda r: (r.center[1], r.center[0]))
            for i, r in enumerate(lst, 1):
                r.index = i

    def ensure_unified_room_naming(self):
        """确保房间采用统一编号命名：type保持基础类型，index为序号"""
        # 首先确保索引正确
        self.ensure_indices()
        
        # 按基础类型分组房间
        by_base_type: dict[str, list[RoomModel]] = {}
        
        for r in self.rooms:
            base_type = r.type  # 现在type已经是基础类型
            by_base_type.setdefault(base_type, []).append(r)
        
        # 为每个基础类型重新分配序号
        for base_type, room_list in by_base_type.items():
            # 按位置排序（从上到下，从左到右）
            room_list.sort(key=lambda r: (r.center[1], r.center[0]))
            
            for i, room in enumerate(room_list, 1):
                # 保持类型为基础类型，更新序号
                room.type = base_type  # 确保类型保持基础类型
                room.index = i
                # 更新房间ID包含显示名称
                display_name = f"{base_type}{i}"
                room.id = f"{display_name}_{i}"
                # 更新显示文本以反映新的类型和编号
                room.text_raw = display_name
    
    def _extract_base_type(self, room_type: str) -> str:
        """提取房间的基础类型，去除数字编号"""
        if not room_type:
            return room_type
        
        # 从后往前找，移除末尾的连续数字
        i = len(room_type) - 1
        while i >= 0 and room_type[i].isdigit():
            i -= 1
        
        # 如果整个字符串都是数字，返回原字符串
        if i < 0:
            return room_type
        
        # 返回去除数字后的基础类型
        return room_type[:i+1]

    def next_label_id(self) -> int:
        self._next_label_seed += 1
        return self._next_label_seed

    def init_label_seed(self):
        max_existing = max([r.label_id for r in self.rooms if r.label_id > 0] + [19])
        self._next_label_seed = max_existing

    def assign_label_for_category(self, cat: str) -> int:
        if cat in self.category_label_map:
            return self.category_label_map[cat]
        # 若已有房间使用该类型，复用其 label
        for r in self.rooms:
            if r.type == cat and r.label_id > 0:
                self.category_label_map[cat] = r.label_id
                return r.label_id
        # 否则自增
        new_id = self.next_label_id()
        self.category_label_map[cat] = new_id
        return new_id

    # ---------- 序列化 ----------
    def to_json_dict(self) -> dict:
        self.ensure_indices()
        meta = {
            "image_width": self.img_w,
            "image_height": self.img_h,
            "rooms_detected": len(self.rooms),
            "output_image": self.image_path,
            "house_orientation": self.house_orientation,
            "north_angle": self.north_angle,
            "magnetic_declination": self.magnetic_declination,
        }
        
        # 添加原图和结果图路径
        if self.original_image_path:
            meta["original_image"] = self.original_image_path
        if self.result_image_path:
            meta["result_image"] = self.result_image_path
        
        return {
            "meta": meta,
            "rooms": [
                {
                    "type": r.type,
                    "index": r.index,
                    "label_id": r.label_id,
                    "center": {"x": r.center[0], "y": r.center[1]},
                    "center_normalized": r.center_normalized,
                    "bbox": {
                        "x1": r.bbox[0],
                        "y1": r.bbox[1],
                        "x2": r.bbox[2],
                        "y2": r.bbox[3],
                        "width": r.bbox[2] - r.bbox[0] + 1,
                        "height": r.bbox[3] - r.bbox[1] + 1,
                    },
                    "area_pixels": r.area_pixels,
                    "text_raw": r.text_raw,
                    "confidence": r.confidence,
                    "distance_to_center": r.distance_to_center,
                    "direction_8": r.direction_8,
                    "source": r.source,
                    "edited": r.edited,
                }
                for r in self.rooms
            ],
        }
