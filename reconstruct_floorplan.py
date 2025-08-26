#!/usr/bin/env python3
"""
从导出的房间JSON结果重建户型示意图
====================================

用途:
 读取 demo_refactored_clean.py 生成的 *_result.json, 基于其中的房间列表与 bbox / center 信息
 重新绘制户型彩色示意图 (近似“分割”效果) + 标注与图例, 便于后续单独做风水/空间分析可视化。

特点:
 1. 使用 utils.rgb_ind_convertor.floorplan_fuse_map_figure 的同一颜色映射, 保证颜色一致性
 2. 支持不同填充模式 (bbox 填充 / 仅轮廓 / 半透明叠加)
 3. 自动生成图例与房间信息摘要
 4. 支持输出 PNG + 可选保存纯遮罩 (label 索引图) 便于后续算法处理

用法示例:
  python reconstruct_floorplan.py output/demo1_result.json
  python reconstruct_floorplan.py output/demo1_result.json --output out_demo1_rebuild.png --fill-mode alpha --grid
  python reconstruct_floorplan.py output/demo1_result.json --mask-out out_demo1_mask.png

JSON结构参考 (rooms 数组中的字段):
  type, index, label_id, center{x,y}, bbox{x1,y1,x2,y2,width,height}, area_pixels,
  text_raw, confidence, distance_to_center, direction_8

注意:
  由于原始 JSON 只保存了 bbox 而不是逐像素 mask, 这里的重建是依据 bbox 的近似示意图, 
  若需要高精度重建需后续在导出阶段加入真实掩膜点集/多边形。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# 颜色映射 (与主流程保持一致)
try:
    from utils.rgb_ind_convertor import floorplan_fuse_map_figure as COLOR_MAP
except Exception:
    COLOR_MAP = {
        0: [255, 255, 255],
        2: [192, 255, 255],
        3: [224, 255, 192],
        4: [255, 224, 128],
        6: [0, 255, 255],
        7: [0, 255, 0],
        8: [224, 224, 128],
        9: [255, 60, 128],
        10: [0, 0, 0],
    }

LABEL_NAMES = {
    0: "背景",
    1: "储物",
    2: "卫生间",
    3: "客厅",
    4: "卧室",
    5: "玄关/大厅",
    6: "阳台",
    7: "厨房",
    8: "书房",
    9: "开口",
    10: "墙体",
}

# ---------------- 中文字体初始化 ----------------
CH_FONT_PATH = None
def _init_ch_font():
    global CH_FONT_PATH
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Source Han Sans CN", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "Sarasa UI SC", "HarmonyOS Sans SC"
    ]
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path and os.path.isfile(path):
                CH_FONT_PATH = path
                # 设置 matplotlib 全局中文支持
                plt.rcParams['font.sans-serif'] = [name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"🈶 使用中文字体: {name} -> {path}")
                return
        except Exception:
            continue
    print("⚠️ 未找到可用中文字体，将可能出现 '?' 乱码，请安装微软雅黑或思源黑体。")

_init_ch_font()


def parse_args():
    p = argparse.ArgumentParser(description="从房间JSON重建户型示意图")
    p.add_argument('json_path', help='输入 *_result.json 路径')
    p.add_argument('--output', '-o', help='输出PNG文件 (默认 <basename>_rebuild.png)')
    p.add_argument('--fill-mode', choices=['bbox', 'outline', 'alpha'], default='bbox',
                   help='房间填充方式: bbox=完全填充, outline=仅边框, alpha=半透明叠加')
    p.add_argument('--alpha', type=float, default=0.55, help='alpha 模式下的透明度 (0-1)')
    p.add_argument('--grid', action='store_true', help='显示网格线')
    p.add_argument('--mask-out', help='同时输出索引mask图 (PNG)')
    p.add_argument('--dpi', type=int, default=160, help='保存分辨率 DPI')
    return p.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"JSON 不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_blank_canvas(meta: Dict[str, Any]) -> np.ndarray:
    w = int(meta.get('image_width', 0) or 0)
    h = int(meta.get('image_height', 0) or 0)
    if w <= 0 or h <= 0:
        raise ValueError('JSON meta 中缺失有效的 image_width/height')
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)  # 白底
    return canvas


def create_label_mask(meta: Dict[str, Any], rooms: List[Dict[str, Any]]) -> np.ndarray:
    mask = np.zeros((meta['image_height'], meta['image_width']), dtype=np.uint8)
    for r in rooms:
        lid = int(r.get('label_id', -1))
        if lid < 0:
            continue
        bbox = r.get('bbox', {})
        x1, y1, x2, y2 = bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2')
        if None in (x1, y1, x2, y2):
            continue
        x1 = max(0, min(mask.shape[1]-1, int(x1)))
        x2 = max(0, min(mask.shape[1]-1, int(x2)))
        y1 = max(0, min(mask.shape[0]-1, int(y1)))
        y2 = max(0, min(mask.shape[0]-1, int(y2)))
        mask[y1: y2+1, x1: x2+1] = lid
    return mask


def apply_rooms(canvas: np.ndarray, rooms: List[Dict[str, Any]], fill_mode: str, alpha: float) -> np.ndarray:
    out = canvas.copy()
    overlay = out.copy()
    for r in rooms:
        lid = int(r.get('label_id', -1))
        if lid not in COLOR_MAP:
            continue
        color = tuple(int(c) for c in COLOR_MAP[lid])  # BGR? COLOR_MAP 是RGB; OpenCV用BGR
        # 转换为 BGR
        color_bgr = (color[2], color[1], color[0])
        bbox = r.get('bbox', {})
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        if fill_mode == 'bbox':
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness=-1)
        elif fill_mode == 'outline':
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness=2)
        elif fill_mode == 'alpha':
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, thickness=-1)
        # 画中心点
        cx, cy = int(r['center']['x']), int(r['center']['y'])
        cv2.circle(out, (cx, cy), 5, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1)
    if fill_mode == 'alpha':
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, dst=out)
    return out


def draw_annotations(img: np.ndarray, rooms: List[Dict[str, Any]]):
    """使用PIL绘制中文，避免 OpenCV putText 出现 '?'。"""
    # 转为 PIL Image (BGR -> RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    if CH_FONT_PATH:
        def get_font(sz):
            try:
                return ImageFont.truetype(CH_FONT_PATH, sz)
            except Exception:
                return ImageFont.load_default()
    else:
        def get_font(sz):
            return ImageFont.load_default()

    for r in rooms:
        lid = int(r.get('label_id', -1))
        name = LABEL_NAMES.get(lid, r.get('type', '房间'))
        bbox = r['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        cx, cy = r['center']['x'], r['center']['y']
        label_text = f"{name}{r.get('index','')}".strip()
        dir8 = r.get('direction_8', '')
        dist = r.get('distance_to_center', 0)
        info_text = f"{dir8} {dist:.0f}px" if dir8 else f"{dist:.0f}px"

        # 绘制边框 (灰色细线)
        draw.rectangle([x1, y1, x2, y2], outline=(40,40,40), width=1)

        # 文字背景与前景
        font_main = get_font(18)
        font_sub = get_font(14)
        # 主标签
        tw, th = draw.textbbox((0,0), label_text, font=font_main)[2:]
        bg_pad = 3
        draw.rectangle([x1+2-bg_pad, y1+2-bg_pad, x1+2+tw+bg_pad, y1+2+th+bg_pad], fill=(0,0,0,160))
        draw.text((x1+2, y1+2), label_text, font=font_main, fill=(255,255,255))
        # 信息标签
        if info_text:
            tw2, th2 = draw.textbbox((0,0), info_text, font=font_sub)[2:]
            by = min(y2 - th2 - 2, y1 + th + 10)
            draw.rectangle([x1+2-bg_pad, by-bg_pad, x1+2+tw2+bg_pad, by+th2+bg_pad], fill=(0,0,0,140))
            draw.text((x1+2, by), info_text, font=font_sub, fill=(255,255,255))

        # 中心点十字
        cross_color = (255,255,255)
        for dx in range(-3,4):
            if 0 <= cx+dx < img.shape[1]:
                pil_img.putpixel((cx+dx, cy), cross_color)
        for dy in range(-3,4):
            if 0 <= cy+dy < img.shape[0]:
                pil_img.putpixel((cx, cy+dy), cross_color)

    # 转回 BGR ndarray
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def build_matplot_figure(rebuilt: np.ndarray, rooms: List[Dict[str, Any]], meta: Dict[str, Any], show_grid: bool, out_path: str, dpi: int):
    h, w = rebuilt.shape[:2]
    fig_w = max(8, w / 150.0)
    fig_h = max(6, h / 150.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(cv2.cvtColor(rebuilt, cv2.COLOR_BGR2RGB))
    ax.set_title(f"重建户型示意图 (房间数: {len(rooms)})", fontsize=14, fontweight='bold')
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    # 图例
    handles = []
    import matplotlib.patches as mpatches
    used_labels = sorted({int(r.get('label_id', -1)) for r in rooms if int(r.get('label_id', -1)) in COLOR_MAP})
    for lid in used_labels:
        rgb = np.array(COLOR_MAP[lid]) / 255.0
        name = LABEL_NAMES.get(lid, str(lid))
        handles.append(mpatches.Patch(color=rgb, label=name))
    if handles:
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=min(6, len(handles)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    data = load_json(args.json_path)
    meta = data.get('meta', {})
    rooms = data.get('rooms', [])
    if not rooms:
        print('⚠️ JSON 中未找到 rooms 数据')
    canvas = build_blank_canvas(meta)

    # 生成索引 mask (先用于可选导出 / 也可未来做更复杂合成)
    mask = create_label_mask(meta, rooms)

    # 填充可视化
    vis = apply_rooms(canvas, rooms, args.fill_mode, args.alpha)
    vis = draw_annotations(vis, rooms)

    # 默认输出名
    out_png = args.output or (str(Path(args.json_path).with_suffix('')) + '_rebuild.png')
    build_matplot_figure(vis, rooms, meta, args.grid, out_png, args.dpi)
    print(f'✅ 重建图已保存: {out_png}')

    if args.mask_out:
        # 将 mask 以调色形式保存 (或灰度)
        color_mask = np.zeros_like(canvas)
        for lid, color in COLOR_MAP.items():
            color_mask[mask == lid] = color
        mask_out_path = args.mask_out
        cv2.imwrite(mask_out_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        print(f'🧩 掩膜图已保存: {mask_out_path}')

    # 输出摘要
    counts = {}
    for r in rooms:
        lid = int(r.get('label_id', -1))
        counts[lid] = counts.get(lid, 0) + 1
    print('📊 房间统计:')
    for lid, cnt in sorted(counts.items()):
        print(f'  - {LABEL_NAMES.get(lid, lid)}: {cnt} 个')
    print('完成。')


if __name__ == '__main__':
    main()
