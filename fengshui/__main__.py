import argparse
import json

from editor.json_io import load_floorplan_json
from . import luoshu_missing_corner as lmc


def main():
    parser = argparse.ArgumentParser(description="Analyze floorplan fengshui missing corners")
    parser.add_argument("json", help="Path to floorplan JSON")
    parser.add_argument("--threshold", type=float, default=0.6, help="Coverage threshold for missing corner")
    parser.add_argument("--output", help="Optional path to save report")
    args = parser.parse_args()

    doc = load_floorplan_json(args.json)
    with open(args.json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    polygon = raw.get("polygon") or raw.get("floor_polygon") or raw.get("outline")
    if not polygon:
        raise ValueError("JSON 缺少 polygon 字段")

    lmc.NORTH_ANGLE = getattr(doc, "north_angle", lmc.NORTH_ANGLE)
    lmc.HOUSE_ORIENTATION = getattr(doc, "house_orientation", lmc.HOUSE_ORIENTATION)

    result = lmc.analyze_missing_corners(polygon, doc.img_w, doc.img_h, threshold=args.threshold)
    lines = [
        f"{item['direction']}方缺角 覆盖率{item['coverage']:.2f} -> {item['suggestion']}" for item in result
    ]
    report = "\n".join(lines) if lines else "无明显缺角"
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
