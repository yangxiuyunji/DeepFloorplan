import argparse
import json
import numpy as np

from editor.json_io import load_floorplan_json
from . import analyze_eightstars, luoshu_missing_corner as lmc, general_guidelines


def create_simple_rectangle_polygon(rooms):
    """从房间数据创建简单的矩形外轮廓（避免凸包算法的过度连接）"""
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
    
    # 使用简单的外接矩形，不进行凸包连接
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    
    # 返回矩形的四个角点
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def create_polygon_from_rooms(rooms):
    """从房间数据创建更精确的外轮廓多边形（与luoshu_visualizer.py保持一致）"""
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze floorplan fengshui by different methods"
    )
    parser.add_argument("json", help="Path to floorplan JSON")
    parser.add_argument(
        "--mode",
        choices=["luoshu", "bazhai"],
        default="luoshu",
        help="Analysis mode: 'luoshu' missing corners or 'bazhai' eight stars",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Coverage threshold for missing corner (luoshu mode)",
    )
    parser.add_argument(
        "--gua",
        help="Personal Ming Gua for BaZhai analysis; if omitted, use the house orientation",
    )
    parser.add_argument("--output", help="Optional path to save report")
    args = parser.parse_args()

    doc = load_floorplan_json(args.json)
    with open(args.json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    polygon = raw.get("polygon") or raw.get("floor_polygon") or raw.get("outline")
    if not polygon:
        # 优先使用简单矩形方法（避免凸包算法的过度连接问题）
        rooms = raw.get("rooms", [])
        polygon = create_simple_rectangle_polygon(rooms)
        
        # 如果用户明确需要凸包方法，可以通过参数控制
        # polygon = create_polygon_from_rooms(rooms)
        
        if not polygon:
            raise ValueError("JSON 缺少 polygon 字段且无法从房间数据创建")

    if args.mode == "luoshu":
        lmc.NORTH_ANGLE = getattr(doc, "north_angle", lmc.NORTH_ANGLE)
        lmc.HOUSE_ORIENTATION = getattr(
            doc, "house_orientation", lmc.HOUSE_ORIENTATION
        )
        
        # 使用修正后的基于房间覆盖率的缺角检测方法
        rooms_data = []
        for room in raw.get("rooms", []):
            bbox = room.get("bbox", {})
            if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                rooms_data.append({"bbox": bbox})
        
        result = lmc.analyze_missing_corners_by_room_coverage(
            rooms_data, doc.img_w, doc.img_h, threshold=args.threshold
        )
        
        lines = [
            f"{item['direction']}方缺角 覆盖率{item['coverage']:.2f} -> {item['suggestion']}"
            for item in result
        ]
        if result:
            lines.append("")
            lines.append("常见化解思路：")
            lines.extend(lmc.general_remedies())
        report = "\n".join(lines) if lines else "无明显缺角"
    else:  # bazhai
        # Use fixed mapping method (same as luoshu_visualizer.py)
        from . import analyze_eightstars_fixed_mapping
        result = analyze_eightstars_fixed_mapping([], doc, gua=args.gua)
        
        lines = []
        lines.append("八宅八星方位分布（固定映射）：")
        for item in result:
            lines.append(f"{item['direction']}: {item['star']} ({item['nature']}) - {item['suggestion']}")
        
        if lines:
            lines.append("")
            lines.append("八宅调整建议：")
            lines.extend(general_guidelines())
        report = "\n".join(lines) if lines else "无方位信息"

    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
