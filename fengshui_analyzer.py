#!/usr/bin/env python3
"""
风水分析工具 - 专门针对户型图进行九宫缺角和八宅八星分析
Usage: python fengshui_analyzer.py [JSON_FILE] [--gua 命卦]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from editor.json_io import load_floorplan_json
from fengshui.bazhai_eightstars import analyze_eightstars, general_guidelines
from fengshui.luoshu_missing_corner import analyze_missing_corners_by_room_coverage, general_remedies


def create_polygon_from_rooms(rooms: List[Dict[str, Any]]) -> List[tuple]:
    """从房间数据创建外轮廓多边形"""
    if not rooms:
        return []
    
    boxes = []
    for room in rooms:
        bbox = room.get("bbox", {})
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            room_type = str(room.get("type", ""))
            if room_type == "阳台":
                w = x2 - x1
                h = y2 - y1
                if abs(w) <= abs(h):
                    cx = (x1 + x2) / 2.0
                    w *= 0.5
                    x1 = cx - w / 2.0
                    x2 = cx + w / 2.0
                else:
                    cy = (y1 + y2) / 2.0
                    h *= 0.5
                    y1 = cy - h / 2.0
                    y2 = cy + h / 2.0
            boxes.append((x1, y1, x2, y2))
    
    if not boxes:
        return []
    
    # 计算包围所有房间的最小外接矩形
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    
    # 返回矩形四个顶点
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def load_floorplan_data(json_path: str) -> tuple:
    """加载户型图数据"""
    try:
        # 使用现有的加载函数
        doc = load_floorplan_json(json_path)
        
        # 同时读取原始JSON数据获取多边形信息
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 尝试获取多边形数据
        polygon = (raw_data.get("polygon") or 
                  raw_data.get("floor_polygon") or 
                  raw_data.get("outline"))
        
        # 如果没有多边形数据，从房间数据创建
        if not polygon:
            rooms = raw_data.get("rooms", [])
            polygon = create_polygon_from_rooms(rooms)
        
        return doc, polygon, raw_data
        
    except Exception as e:
        print(f"加载文件失败: {e}")
        sys.exit(1)


def analyze_luoshu_missing_corners(rooms: List[Dict], width: int, height: int, 
                                 north_angle: int = 90, threshold: float = 0.75) -> Dict[str, Any]:
    """九宫缺角分析"""
    print("=== 九宫缺角分析 ===")
    
    # 执行基于房间覆盖率的分析
    missing_corners = analyze_missing_corners_by_room_coverage(
        rooms, width, height, threshold, north_angle
    )
    
    result = {
        "method": "九宫缺角分析",
        "total_missing": len(missing_corners),
        "missing_corners": missing_corners,
        "remedies": general_remedies()
    }
    
    # 打印结果
    if missing_corners:
        print(f"发现 {len(missing_corners)} 个缺角:")
        for corner in missing_corners:
            print(f"  {corner['direction']}方缺角 - 覆盖率: {corner['coverage']:.2%}")
            print(f"    建议: {corner['suggestion']}")
        
        print("\n通用化解思路:")
        for i, remedy in enumerate(general_remedies(), 1):
            print(f"  {i}. {remedy}")
    else:
        print("未发现明显缺角，户型较为方正。")
    
    return result


def analyze_bazhai_eightstars_method(polygon: List[tuple], rooms: List[Dict], 
                                   doc, gua: str = None) -> Dict[str, Any]:
    """八宅八星分析 - 适配新架构：type为基础类型，index为序号"""
    print("\n=== 八宅八星分析 ===")
    
    # 准备房间数据，现在type已经是基础类型，index是序号
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
    
    # 执行分析
    star_analysis = analyze_eightstars(polygon, room_data, doc, gua)
    
    # 为分析结果添加额外信息
    enhanced_analysis = []
    for i, analysis in enumerate(star_analysis):
        room_info = room_data[i]
        enhanced_item = {
            "room_id": analysis["room"],  # 显示名称，如"卧室1"
            "room_type": room_info["type"],  # 基础类型，如"卧室"
            "direction": analysis["direction"],
            "star": analysis.get("star", ""),
            "nature": analysis.get("nature", ""),
            "suggestion": analysis.get("suggestion", "根据方位合理布置。"),
            "center_position": f"({room_info['center'][0]:.0f}, {room_info['center'][1]:.0f})",
            "area_pixels": room_info["area_pixels"],
            "room_index": room_info["index"]
        }
        enhanced_analysis.append(enhanced_item)
    
    result = {
        "method": "八宅八星分析",
        "gua": gua or f"根据房屋朝向({getattr(doc, 'house_orientation', '坐北朝南')})",
        "total_rooms": len(enhanced_analysis),
        "room_analysis": enhanced_analysis,
        "guidelines": general_guidelines()
    }
    
    # 打印结果
    if gua:
        print(f"使用命卦: {gua}")
    else:
        print(f"使用房屋朝向: {getattr(doc, 'house_orientation', '坐北朝南')}")
    
    print(f"\n各房间星位分析 (共{len(enhanced_analysis)}个房间):")
    
    # 按吉凶分类显示
    auspicious = []  # 吉星
    inauspicious = []  # 凶星
    
    for analysis in enhanced_analysis:
        if analysis.get("nature") == "吉":
            auspicious.append(analysis)
        elif analysis.get("nature") == "凶":
            inauspicious.append(analysis)
    
    if auspicious:
        print("\n吉星方位:")
        for item in auspicious:
            print(f"  {item['room_id']} - {item['direction']}方 - {item['star']}星")
            print(f"    位置: {item['center_position']}, 面积: {item['area_pixels']}像素")
            print(f"    建议: {item['suggestion']}")
    
    if inauspicious:
        print("\n凶星方位:")
        for item in inauspicious:
            print(f"  {item['room_id']} - {item['direction']}方 - {item['star']}星")
            print(f"    位置: {item['center_position']}, 面积: {item['area_pixels']}像素")
            print(f"    建议: {item['suggestion']}")
    
    # 显示其他房间
    others = [item for item in enhanced_analysis if item.get("nature") not in ["吉", "凶"]]
    if others:
        print("\n其他房间:")
        for item in others:
            star_info = f" - {item['star']}星" if item['star'] else ""
            print(f"  {item['room_id']} - {item['direction']}方{star_info}")
            print(f"    位置: {item['center_position']}")
    
    print("\n八宅布局调整建议:")
    for i, guideline in enumerate(general_guidelines(), 1):
        print(f"  {i}. {guideline}")
    
    return result


def save_analysis_report(luoshu_result: Dict, bazhai_result: Dict, output_path: str):
    """保存分析报告"""
    report = {
        "analysis_date": "2025年9月2日",
        "luoshu_analysis": luoshu_result,
        "bazhai_analysis": bazhai_result,
        "summary": {
            "missing_corners_count": luoshu_result["total_missing"],
            "rooms_analyzed": bazhai_result["total_rooms"],
            "main_issues": [],
            "key_recommendations": []
        }
    }
    
    # 添加主要问题汇总
    if luoshu_result["total_missing"] > 0:
        missing_directions = [corner["direction"] for corner in luoshu_result["missing_corners"]]
        report["summary"]["main_issues"].append(f"发现缺角: {', '.join(missing_directions)}方")
    
    # 添加凶星方位的房间汇总
    inauspicious_rooms = []
    for room_analysis in bazhai_result["room_analysis"]:
        if room_analysis.get("nature") == "凶":
            inauspicious_rooms.append(f"{room_analysis['room_id']}({room_analysis['star']}星)")
    
    if inauspicious_rooms:
        report["summary"]["key_recommendations"].append(f"重点关注凶星方位: {', '.join(inauspicious_rooms)}")
    
    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析报告已保存至: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="户型图风水分析工具")
    parser.add_argument("json", help="户型图JSON文件路径")
    parser.add_argument("--gua", help="个人命卦 (如: 坎, 震, 巽, 离, 坤, 乾, 兑, 艮)")
    parser.add_argument("--threshold", type=float, default=0.75, 
                       help="缺角判定阈值 (默认: 0.75)")
    parser.add_argument("--output", help="保存分析报告的路径")
    
    args = parser.parse_args()
    
    print(f"正在分析户型图: {args.json}")
    print("=" * 50)
    
    # 加载数据
    doc, polygon, raw_data = load_floorplan_data(args.json)
    meta = raw_data.get("meta", {})
    width = meta.get("image_width", 800)
    height = meta.get("image_height", 600)
    north_angle = getattr(doc, "north_angle", 90)
    
    print(f"房屋信息:")
    print(f"  朝向: {getattr(doc, 'house_orientation', '坐北朝南')}")
    print(f"  北向角度: {north_angle}°")
    print(f"  图像尺寸: {width}×{height}")
    print(f"  房间数量: {len(doc.rooms)}")
    
    # 执行九宫缺角分析
    luoshu_result = analyze_luoshu_missing_corners(
        raw_data.get("rooms", []), width, height, north_angle, args.threshold
    )
    
    # 执行八宅八星分析  
    bazhai_result = analyze_bazhai_eightstars_method(
        polygon, raw_data.get("rooms", []), doc, args.gua
    )
    
    # 保存报告
    if args.output:
        save_analysis_report(luoshu_result, bazhai_result, args.output)
    else:
        # 默认保存在同目录下
        json_path = Path(args.json)
        output_path = json_path.parent / f"{json_path.stem}_fengshui_analysis.json"
        save_analysis_report(luoshu_result, bazhai_result, str(output_path))


if __name__ == "__main__":
    main()
