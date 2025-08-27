# -*- coding: utf-8 -*-
"""
通用 OCR 完整识别脚本：对指定平面图运行 PaddleOCR (中文) 主动输出：
  1) 全部检测 + 识别结果 JSON (含 polygon, bbox, text, score)
  2) 可视化 PNG (编号 + 文本 + 置信度)
  3) 统计摘要：
       - 总框数 / 有效文本数
       - 关键候选(厨房/卧室/卫/餐/阳)匹配情况
       - 小尺寸疑似单字后缀框数量
  4) 若首轮未发现 "厨" 且配置允许，自动触发一次放宽参数的第二轮识别并合并结果

用法:  python ocr_full_demo3.py [image_path] [--no-second-pass]
默认图片: demo/demo3.jpg
输出: output/ocr_full_<basename>.json / _vis.png / _summary.txt

注意: 为避免 Windows 控制台编码问题，不输出 emoji。
"""
import os, sys, json, math
from pathlib import Path
import argparse
import cv2
import numpy as np
from paddleocr import PaddleOCR

# -------------------- 参数解析 --------------------
parser = argparse.ArgumentParser()
parser.add_argument('image', nargs='?', default='demo/demo3.jpg')
parser.add_argument('--no-second-pass', action='store_true', help='禁用缺失厨字时的第二次放宽参数识别')
parser.add_argument('--out-dir', default='output', help='输出目录')
parser.add_argument('--min-text-len', type=int, default=1, help='最少文本长度过滤 (识别后)')
args = parser.parse_args()
img_path = Path(args.image)
if not img_path.exists():
    print(f"输入图片不存在: {img_path}")
    sys.exit(1)

out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
base_name = img_path.stem

# -------------------- OCR 初始化 --------------------
# 主通道参数 (与主流程接近)
primary_params = dict(det=True, rec=True, use_angle_cls=True, lang='ch',
                      det_db_thresh=0.2, det_db_box_thresh=0.4, drop_score=0.3,
                      det_db_unclip_ratio=2.5, use_dilation=True)

# 放宽通道参数 (更低阈值, 去 angle, 更易出小框)
relaxed_params = dict(det=True, rec=True, use_angle_cls=False, lang='ch',
                      det_db_thresh=0.18, det_db_box_thresh=0.35, drop_score=0.20,
                      det_db_unclip_ratio=2.7, use_dilation=True)

print("加载 PaddleOCR 主通道参数 ...")
ocr_primary = PaddleOCR(**primary_params)
print("主通道加载完成")
ocr_relaxed = None
if not args.no_second_pass:
    print("预加载放宽通道参数(备用)...")
    ocr_relaxed = PaddleOCR(**relaxed_params)
    print("放宽通道加载完成")

# -------------------- OCR 运行函数 --------------------

def run_ocr(ocr_engine, image_path: Path):
    result = ocr_engine.ocr(str(image_path), cls=True)
    if not result:
        return []
    # 兼容多页 list[ [ ( (quad), (text,score) ), ... ] ]
    page = result[0]
    out = []
    for item in page:
        if not item or len(item) < 2:
            continue
        quad = item[0]
        txt = item[1][0] if len(item[1])>0 else ''
        score = float(item[1][1]) if len(item[1])>1 else 0.0
        quad_np = np.array(quad, dtype=np.float32)
        minx = float(quad_np[:,0].min()); maxx = float(quad_np[:,0].max())
        miny = float(quad_np[:,1].min()); maxy = float(quad_np[:,1].max())
        out.append({
            'quad': quad_np.tolist(),
            'bbox': [int(minx), int(miny), int(maxx), int(maxy)],
            'text': txt,
            'score': score,
            'w': int(maxx-minx+1), 'h': int(maxy-miny+1)
        })
    return out

# -------------------- 主通道执行 --------------------
print("执行主通道 OCR ...")
primary_res = run_ocr(ocr_primary, img_path)
print(f"主通道返回框数: {len(primary_res)}")

# 过滤空文本 / 长度
primary_res = [r for r in primary_res if r['text'] and len(r['text']) >= args.min_text_len]
print(f"主通道有效文本框数(过滤后): {len(primary_res)}")

texts_concat = ''.join(r['text'] for r in primary_res)
has_kitchen = ('厨' in texts_concat) or ('灶' in texts_concat) or ('厨房' in texts_concat)
need_second = (not has_kitchen) and (not args.no_second_pass)

relaxed_res = []
if need_second:
    print("未检出任何厨/灶字样, 触发放宽通道 ...")
    relaxed_res = run_ocr(ocr_relaxed, img_path)
    before_filter = len(relaxed_res)
    relaxed_res = [r for r in relaxed_res if r['text'] and len(r['text'])>=args.min_text_len]
    print(f"放宽通道返回框数: {before_filter}, 有效文本框数: {len(relaxed_res)}")
else:
    if has_kitchen:
        print("主通道已包含厨房相关字样, 不触发放宽通道")
    else:
        print("未加载放宽通道或被禁止第二次识别")

# -------------------- 合并去重 --------------------
# 简单基于文本+IOU 的合并

def iou(b1, b2):
    x1 = max(b1['bbox'][0], b2['bbox'][0])
    y1 = max(b1['bbox'][1], b2['bbox'][1])
    x2 = min(b1['bbox'][2], b2['bbox'][2])
    y2 = min(b1['bbox'][3], b2['bbox'][3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2-x1+1)*(y2-y1+1)
    a1 = (b1['bbox'][2]-b1['bbox'][0]+1)*(b1['bbox'][3]-b1['bbox'][1]+1)
    a2 = (b2['bbox'][2]-b2['bbox'][0]+1)*(b2['bbox'][3]-b2['bbox'][1]+1)
    return inter / float(a1 + a2 - inter + 1e-6)

merged = []
for cand in primary_res + relaxed_res:
    dup = False
    for m in merged:
        if cand['text'] == m['text'] and iou(cand, m) > 0.5:
            # 保留高分
            if cand['score'] > m['score']:
                m.update(cand)
            dup = True
            break
    if not dup:
        merged.append(cand)

print(f"合并后总文本框数: {len(merged)}")

# -------------------- 统计分析 --------------------
key_tokens = ['厨','厨房','灶','卧','卧室','厅','客厅','卫','卫生间','浴','阳','阳台','餐','餐厅']
found_tokens = {k:0 for k in key_tokens}
for r in merged:
    for k in key_tokens:
        if k in r['text']:
            found_tokens[k] += 1

small_suffix_boxes = [r for r in merged if r['w']<=30 and r['h']<=30 and len(r['text'])==1]

# 排序(按置信度)
merged_sorted = sorted(merged, key=lambda x: x['score'], reverse=True)

summary_lines = []
summary_lines.append(f"图片: {img_path} 尺寸统计: ")
try:
    img0 = cv2.imread(str(img_path))
    if img0 is not None:
        H,W = img0.shape[:2]
        summary_lines.append(f"  图像宽高: {W}x{H}")
except Exception:
    pass
summary_lines.append(f"主通道文本框: {len(primary_res)} 放宽通道文本框: {len(relaxed_res)} 合并后: {len(merged)}")
summary_lines.append("关键字出现统计:")
for k,v in found_tokens.items():
    if v>0:
        summary_lines.append(f"  {k}: {v}")
summary_lines.append(f"疑似单字符后缀框(<=30x30 且长度1): {len(small_suffix_boxes)}")
if need_second and len(relaxed_res)==0:
    summary_lines.append("放宽通道仍未发现任何文本 (可能图片中字体极淡或区域遮挡)")
if not any(found_tokens[k] for k in ['厨','厨房','灶']):
    summary_lines.append("未发现厨房相关字, 需要基于结构/连通域启发式")

# Top 20 文本
summary_lines.append("前20高置信度文本:")
for r in merged_sorted[:20]:
    summary_lines.append(f"  {r['text']} (score={r['score']:.3f} bbox={r['bbox']})")

summary_txt_path = out_dir / f"ocr_full_{base_name}_summary.txt"
with open(summary_txt_path,'w',encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f"已生成摘要: {summary_txt_path}")

# -------------------- 可视化 --------------------
vis_img = cv2.imread(str(img_path))
if vis_img is None:
    print("无法读取图像进行可视化")
else:
    for idx, r in enumerate(merged_sorted):
        quad_np = np.array(r['quad'], dtype=np.int32)
        cv2.polylines(vis_img, [quad_np], True, (0,0,255), 1, cv2.LINE_AA)
        label = f"{idx}:{r['text']}:{r['score']:.2f}"
        x,y = r['bbox'][0], r['bbox'][1]-2
        cv2.putText(vis_img, label, (x, max(10,y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
    vis_path = out_dir / f"ocr_full_{base_name}_vis.png"
    cv2.imwrite(str(vis_path), vis_img)
    # 打印可视化路径
    print(f"已保存可视化: {vis_path}")

# -------------------- JSON 输出 --------------------
json_path = out_dir / f"ocr_full_{base_name}.json"
with open(json_path,'w',encoding='utf-8') as f:
    json.dump({
        'image': str(img_path),
        'results': merged_sorted,
        'primary_count': len(primary_res),
        'relaxed_count': len(relaxed_res),
        'found_tokens': found_tokens,
        'single_char_small_boxes': len(small_suffix_boxes)
    }, f, ensure_ascii=False, indent=2)
print(f"已保存 JSON: {json_path}")

print("完成。")
