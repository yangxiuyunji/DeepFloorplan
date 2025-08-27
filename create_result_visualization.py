#!/usr/bin/env python3
"""根据 *_result.json 生成房间示意图 (仅基于 JSON 元信息, 不依赖原预测像素)。

用法:
  python create_result_visualization.py output/demo2_result.json
输出:
  在同目录生成 demo2_result_rebuild.png (若已存在则覆盖)

特点:
  - 按 label_id 上色 (使用 floorplan_fuse_map_figure)
  - 画出房间 bounding box 半透明填充 + 边框
  - 标注: 类型(index)  方向  面积
  - 标出中心点 (十字)
  - 自动生成图例
"""
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

# 直接复用颜色 (复制避免导入整个推理依赖)
FLOORPLAN_FUSE_MAP_FIGURE: Dict[int, Tuple[int,int,int]] = {
    0:(255,255,255), 1:(192,192,224), 2:(192,255,255), 3:(224,255,192), 4:(255,224,128),
    5:(255,160,96), 6:(0,255,255), 7:(0,255,0), 8:(224,224,128), 9:(255,60,128), 10:(0,0,0)
}

ROOM_NAME_ALIAS = { # 仅用于图例排序
    1:"柜/储物", 2:"卫生间", 3:"客厅/餐厨", 4:"卧室", 5:"过道", 6:"阳台", 7:"厨房", 8:"备用", 9:"门窗", 10:"墙线"
}

def pick_font(size=16):
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except Exception:
        try:
            return ImageFont.truetype("simhei.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_cross(draw: ImageDraw.ImageDraw, x:int, y:int, color=(0,0,0)):
    draw.line((x-4,y, x+4,y), fill=color, width=1)
    draw.line((x,y-4, x,y+4), fill=color, width=1)


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def visualize(json_path: Path):
    data = load_json(json_path)
    meta = data['meta']
    W = meta['image_width']; H = meta['image_height']
    rooms = data.get('rooms', [])

    # 背景
    img = Image.new('RGB', (W, H), (255,255,255))
    draw = ImageDraw.Draw(img, 'RGBA')
    font = pick_font(18)
    small_font = pick_font(14)

    # 按面积从小到大绘制 (小的在上层)
    rooms_sorted = sorted(rooms, key=lambda r: r.get('area_pixels', 0))

    for r in rooms_sorted:
        lid = int(r.get('label_id', 0))
        color = FLOORPLAN_FUSE_MAP_FIGURE.get(lid, (200,200,200))
        bbox = r['bbox']
        x1,y1,x2,y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        # 半透明填充
        fill_color = (*color, 80)
        draw.rectangle([x1,y1,x2,y2], fill=fill_color, outline=tuple(color), width=2)
        # 中心
        cx,cy = r['center']['x'], r['center']['y']
        draw_cross(draw, cx, cy, color=(0,0,0))
        # 文本
        txt = f"{r['type']}{r['index']}  {r['direction_8']}  A={r['area_pixels']}"
        try:
            tw,th = draw.textsize(txt, font=small_font)  # Pillow<10
        except Exception:
            # Pillow>=10 用 textbbox
            bbox_txt = draw.textbbox((0,0), txt, font=small_font)
            tw,th = bbox_txt[2]-bbox_txt[0], bbox_txt[3]-bbox_txt[1]
        tx = min(max(x1, cx - tw//2), W - tw - 2)
        ty = min(max(y1, cy - th//2), H - th - 2)
        draw.rectangle([tx-2, ty-1, tx+tw+2, ty+th+1], fill=(255,255,255,190))
        draw.text((tx,ty), txt, font=small_font, fill=(0,0,0))

    # 图例
    legend_items = sorted({(lid, ROOM_NAME_ALIAS.get(lid, str(lid))) for lid in [r.get('label_id',0) for r in rooms_sorted]})
    lx, ly = 10, 10
    for lid, name in legend_items:
        c = FLOORPLAN_FUSE_MAP_FIGURE.get(int(lid),(0,0,0))
        draw.rectangle([lx, ly, lx+26, ly+20], fill=(*c,255), outline=(0,0,0))
        draw.text((lx+32, ly+2), f"{name}({lid})", font=small_font, fill=(0,0,0))
        ly += 24

    out_path = json_path.parent / (json_path.stem.replace('_result','_result_rebuild') + '.png')
    img.save(out_path)
    print(f"✅ 生成: {out_path}")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("用法: python create_result_visualization.py output/demo2_result.json")
        return 1
    jp = Path(sys.argv[1])
    if not jp.is_file():
        print(f"未找到文件: {jp}")
        return 2
    visualize(jp)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
