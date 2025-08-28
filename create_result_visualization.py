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

# 与推理保持一致：直接导入 utils.rgb_ind_convertor 共享 color map
try:
    from utils.rgb_ind_convertor import floorplan_fuse_map_figure as FLOORPLAN_FUSE_MAP_FIGURE  # type: ignore
except Exception:
    # 兜底：若导入失败，使用旧备份（不建议长期依赖）
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
def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """兼容 Pillow 不同版本的文字尺寸测量。"""
    if hasattr(draw, "textbbox"):
        try:
            if hasattr(draw, "textsize"):
                return draw.textsize(text, font=font)  # type: ignore[attr-defined]
        except Exception:
            pass
        bbox = draw.textbbox((0,0), text, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    return draw.textsize(text, font=font)  # type: ignore[attr-defined]


def draw_cross(draw: ImageDraw.ImageDraw, x:int, y:int, color=(0,0,0)):
    draw.line((x-4,y, x+4,y), fill=color, width=1)
    draw.line((x,y-4, x,y+4), fill=color, width=1)


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box: Tuple[int,int,int,int], color=(255,0,0), dash=6, gap=4, width=1):
    x1,y1,x2,y2 = box
    # top
    x = x1
    while x < x2:
        x_end = min(x + dash, x2)
        draw.line((x,y1,x_end,y1), fill=color, width=width)
        x += dash + gap
    # bottom
    x = x1
    while x < x2:
        x_end = min(x + dash, x2)
        draw.line((x,y2,x_end,y2), fill=color, width=width)
        x += dash + gap
    # left
    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        draw.line((x1,y,x1,y_end), fill=color, width=width)
        y += dash + gap
    # right
    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        draw.line((x2,y,x2,y_end), fill=color, width=width)
        y += dash + gap


def visualize(json_path: Path, scale: float = 1.8):
    data = load_json(json_path)
    meta = data['meta']
    W = meta['image_width']; H = meta['image_height']
    rooms = data.get('rooms', [])
    # 右侧扩展画布用于图例，保持主图 (0,0) 坐标系不变
    LEGEND_PAD = 30
    LEGEND_WIDTH = 260
    OUT_W = W + LEGEND_PAD + LEGEND_WIDTH
    OUT_H = H
    img = Image.new('RGB', (OUT_W, OUT_H), (255,255,255))
    draw = ImageDraw.Draw(img, 'RGBA')
    font = pick_font(18)
    small_font = pick_font(14)

    # 按面积从小到大绘制 (小的在上层)
    rooms_sorted = sorted(rooms, key=lambda r: r.get('area_pixels', 0))

    # 已放置标签占用矩形列表，用于简单避让
    placed_label_rects = []  # (x1,y1,x2,y2)

    def place_label(box, tw, th):
        x1,y1,x2,y2 = box
        margin = 12
        # 初始尝试：放在左上外侧
        tx = x1
        ty = y1 - th - margin
        placement = 'top'
        if ty < 0:
            # 放底部左侧
            ty = y2 + margin
            placement = 'bottom'
        # 边界裁剪横向
        if tx + tw > W:
            tx = W - tw - 1
        if tx < 0:
            tx = 0
        # 简单避让：如果与已有标签重叠则垂直位移
        def overlaps(r1,r2):
            return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])
        moved = False
        while True:
            rect = (tx, ty, tx+tw, ty+th)
            conflict = False
            for pr in placed_label_rects:
                if overlaps(rect, pr):
                    conflict = True
                    break
            if not conflict:
                placed_label_rects.append(rect)
                return tx, ty, placement
            # 冲突则沿垂直方向挪动
            moved = True
            shift = th + 4
            if placement == 'top':
                ty -= shift
                if ty < 0:
                    # 改为底部模式
                    placement = 'bottom'
                    ty = y2 + margin
            else:
                ty += shift
                if ty + th > H:
                    # 回到顶部再尝试
                    placement = 'top'
                    ty = y1 - th - margin
            # 防止死循环（极端情况下放回原始并强制添加）
            if not (0 - 3*H < ty < 3*H):
                placed_label_rects.append(rect)
                return tx, max(0,min(ty,H-th-1)), placement

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
        # 文本内部放置：始终使用两行，按需缩小字体直到适配
        line1 = f"{r['type']}{r['index']}  {r['direction_8']}"  # 第一行：类型+方向
        line2_full = f"A={r['area_pixels']} ({x1},{y1})-({x2},{y2})"  # 第二行：面积+坐标
        max_font = 22
        min_font = 8
        fitted_font = pick_font(min_font)
        final_line2 = line2_full
        for fs in range(max_font, min_font-1, -1):
            f_try = pick_font(fs)
            w1,h1 = measure_text(draw, line1, f_try)
            w2,h2 = measure_text(draw, line2_full, f_try)
            tw = max(w1,w2); th = h1 + h2 + 2
            if tw + 10 <= (x2 - x1) and th + 10 <= (y2 - y1):
                fitted_font = f_try
                break
        else:
            # 仍放不下：逐步截短第二行坐标部分
            base = f"A={r['area_pixels']}"
            for fs in range(max_font, min_font-1, -1):
                f_try = pick_font(fs)
                shortened = base
                w1,h1 = measure_text(draw, line1, f_try)
                w2,h2 = measure_text(draw, shortened, f_try)
                tw = max(w1,w2); th = h1 + h2 + 2
                if tw + 10 <= (x2 - x1) and th + 10 <= (y2 - y1):
                    fitted_font = f_try
                    final_line2 = shortened
                    break
        # 重新测量用最终字体
        w1,h1 = measure_text(draw, line1, fitted_font)
        w2,h2 = measure_text(draw, final_line2, fitted_font)
        tw = max(w1,w2); th = h1 + h2 + 2
        inner_w = x2 - x1; inner_h = y2 - y1
        tx = x1 + (inner_w - tw)//2
        ty = y1 + (inner_h - th)//2
        pad = 4
        bx1 = int(tx - pad); by1 = int(ty - pad)
        bx2 = int(tx + tw + pad); by2 = int(ty + th + pad)
        bx1 = max(x1+1, bx1); by1 = max(y1+1, by1)
        bx2 = min(x2-1, bx2); by2 = min(y2-1, by2)
        draw.rectangle([bx1,by1,bx2,by2], fill=(255,255,255,210))
        draw_dashed_rect(draw, (bx1,by1,bx2,by2), color=(255,0,0), dash=6, gap=4, width=1)
        draw.text(( (bx1+bx2 - w1)//2, ty ), line1, font=fitted_font, fill=(0,0,0))
        draw.text(( (bx1+bx2 - w2)//2, ty + h1 + 2 ), final_line2, font=fitted_font, fill=(0,0,0))

    # 坐标轴（保持原始坐标系）：在主图区域绘制 X/Y 轴及刻度
    axis_color = (0,0,0,255)
    # X 轴 (顶端) 与 Y 轴 (左端)
    draw.line((0,0, W,0), fill=axis_color, width=1)
    draw.line((0,0, 0,H), fill=axis_color, width=1)
    tick_font = pick_font(12)
    tick_step = max(50, (W//12)//10*10)  # 动态但不太密；向下取整到10倍
    for x in range(0, W, tick_step):
        draw.line((x,0, x,6), fill=axis_color, width=1)
        txt = str(x)
        tw,th = measure_text(draw, txt, tick_font)
        if x+tw < W:
            draw.text((x,6), txt, font=tick_font, fill=(0,0,0))
    for y in range(0, H, tick_step):
        draw.line((0,y, 6,y), fill=axis_color, width=1)
        txt = str(y)
        tw,th = measure_text(draw, txt, tick_font)
        if y+th < H:
            draw.text((6,y), txt, font=tick_font, fill=(0,0,0))

    # 图例：放在右侧扩展区域
    legend_items = sorted({(lid, ROOM_NAME_ALIAS.get(lid, str(lid))) for lid in [r.get('label_id',0) for r in rooms_sorted]})
    lx = W + LEGEND_PAD
    ly = 20
    draw.text((lx, 0), "颜色图例 / Legend (与识别一致)", font=font, fill=(0,0,0))
    for lid, name in legend_items:
        c = FLOORPLAN_FUSE_MAP_FIGURE.get(int(lid),(0,0,0))
        draw.rectangle([lx, ly, lx+36, ly+24], fill=(*c,255), outline=(0,0,0))
        rgb_txt = f"RGB{tuple(c)}"
        draw.text((lx+44, ly+2), f"{name} (id={lid}) {rgb_txt}", font=small_font, fill=(0,0,0))
        ly += 30

    # 可选整体放大（后期放大防止重算所有坐标）。scale>1 时双三次放大。
    if scale and scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.BICUBIC)
    out_path = json_path.parent / (json_path.stem.replace('_result','_result_rebuild') + '.png')
    img.save(out_path)
    print(f"✅ 生成: {out_path}")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("用法: python create_result_visualization.py <json_path> [--scale 2.0]")
        return 1
    # 简易参数解析
    args = sys.argv[1:]
    scale = 1.0
    if '--scale' in args:
        i = args.index('--scale')
        if i+1 < len(args):
            try:
                scale = float(args[i+1])
            except ValueError:
                pass
            # 去掉参数避免被当作路径
            del args[i:i+2]
    jp = Path(args[0])
    if not jp.is_file():
        print(f"未找到文件: {jp}")
        return 2
    visualize(jp, scale=scale)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
