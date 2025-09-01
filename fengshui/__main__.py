import argparse
import json

from editor.json_io import load_floorplan_json
from . import analyze_eightstars, luoshu_missing_corner as lmc, general_guidelines


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
        raise ValueError("JSON 缺少 polygon 字段")

    if args.mode == "luoshu":
        lmc.NORTH_ANGLE = getattr(doc, "north_angle", lmc.NORTH_ANGLE)
        lmc.HOUSE_ORIENTATION = getattr(
            doc, "house_orientation", lmc.HOUSE_ORIENTATION
        )
        result = lmc.analyze_missing_corners(
            polygon, doc.img_w, doc.img_h, threshold=args.threshold
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
        rooms = [{"bbox": r.bbox, "name": r.type} for r in doc.rooms]
        result = analyze_eightstars(polygon, rooms, doc, gua=args.gua)
        lines = [
            f"{item['room']} {item['direction']} {item['star']} {item['nature']} {item['suggestion']}"
            for item in result
        ]
        if lines:
            lines.append("")
            lines.append("八宅调整建议：")
            lines.extend(general_guidelines())
        report = "\n".join(lines) if lines else "无房间信息"

    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
