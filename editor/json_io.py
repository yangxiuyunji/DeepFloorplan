import json, uuid
from pathlib import Path
from .models import FloorplanDocument, RoomModel
from typing import Optional

DEFAULT_BASE_CATEGORIES = ["厨房", "卫生间", "客厅", "卧室", "阳台", "书房"]

def extract_base_type(room_type: str) -> str:
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

def load_floorplan_json(path: str) -> FloorplanDocument:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    rooms = data.get("rooms", [])
    doc = FloorplanDocument(json_path=path)
    w = meta.get("image_width")
    h = meta.get("image_height")
    if not w or not h:
        raise ValueError("JSON 缺少 image_width / image_height")
    doc.set_image_meta(int(w), int(h))
    doc.image_path = meta.get("output_image")  # 可能为空
    
    # 加载原图和结果图路径
    doc.original_image_path = meta.get("original_image")
    doc.result_image_path = meta.get("result_image")
    
    # 全局属性
    if "house_orientation" in meta:
        doc.house_orientation = meta.get("house_orientation") or doc.house_orientation
    if "north_angle" in meta:
        try:
            doc.north_angle = int(meta.get("north_angle"))
        except Exception:
            pass
    categories = set(DEFAULT_BASE_CATEGORIES)
    for r in rooms:
        room_type = r.get("type", "未知")
        # room.type现在已经是基础类型，直接添加即可
        categories.add(room_type)
    doc.categories = list(categories)
    for r in rooms:
        bbox_dict = r.get("bbox", {})
        x1 = bbox_dict.get("x1")
        y1 = bbox_dict.get("y1")
        x2 = bbox_dict.get("x2")
        y2 = bbox_dict.get("y2")
        if None in (x1, y1, x2, y2):
            continue
        
        # 生成稳定的 ID：基于类型和索引，确保相同房间有相同 ID
        room_type = r.get("type", "未知")
        room_index = r.get("index", 1)
        stable_id = f"{room_type}_{room_index}"
        
        rm = RoomModel(
            id=r.get("id") or stable_id,
            type=room_type,
            label_id=int(r.get("label_id", -1)),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            text_raw=r.get("text_raw", ""),
            confidence=float(r.get("confidence", 0.0)),
            source=r.get("source", "ocr"),
            edited=bool(r.get("edited", False)),
        )
        rm.index = room_index  # 确保索引正确设置
        rm.recompute(doc.img_w, doc.img_h)
        doc.add_room(rm)
    doc.init_label_seed()
    return doc

def save_floorplan_json(doc: FloorplanDocument, path: Optional[str] = None) -> str:
    if path is None:
        # 默认写回同路径 _edited.json
        base = Path(doc.json_path or doc.output_path or "edited_result.json")
        if not base.name.endswith("_edited.json"):
            path = str(base.with_name(base.stem + "_edited.json"))
        else:
            path = str(base)
    
    # 在保存前确保统一编号
    doc.ensure_unified_room_naming()
    
    data = doc.to_json_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
