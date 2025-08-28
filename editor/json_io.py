import json, uuid
from pathlib import Path
from .models import FloorplanDocument, RoomModel

DEFAULT_BASE_CATEGORIES = ["厨房", "卫生间", "客厅", "卧室", "阳台", "书房"]

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
        categories.add(r.get("type", "未知"))
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

def save_floorplan_json(doc: FloorplanDocument, path: str | None = None) -> str:
    if path is None:
        # 默认写回同路径 _edited.json
        base = Path(doc.json_path or doc.output_path or "edited_result.json")
        if not base.name.endswith("_edited.json"):
            path = str(base.with_name(base.stem + "_edited.json"))
        else:
            path = str(base)
    data = doc.to_json_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
