# -*- coding: utf-8 -*-
"""
纯检测(仅 det，不做识别)脚本：输出 demo/demo2.jpg 的所有检测框，方便判断是否存在潜在的 C / B 单字符框。
不修改主工程代码，可独立运行。
生成：
  output/ocr_det_only_boxes.json  (包含全部 polygon/quad 框坐标)
  output/ocr_det_only_vis.png     (简单可视化)
用法：
  python ocr_det_only_demo2.py [可选: 图片路径]
"""
import json, sys, os
from pathlib import Path
import cv2
import numpy as np
from paddleocr import PaddleOCR

IMG_PATH = Path(sys.argv[1]) if len(sys.argv)>1 else Path('demo/demo2.jpg')
if not IMG_PATH.exists():
    print(f"❌ 输入图片不存在: {IMG_PATH}")
    sys.exit(1)

# 仅检测模型
ocr = PaddleOCR(det=True, rec=False, use_angle_cls=False, lang='ch', det_db_thresh=0.2, det_db_box_thresh=0.4, det_db_unclip_ratio=2.5, use_dilation=True)
print("✅ PaddleOCR 检测模型加载完成 (rec 关闭)")

result = ocr.ocr(str(IMG_PATH), cls=False)
# result: list[ [ (box, text, conf) ... ] ] 但 rec=False 时第二第三项为空, 只保留 box
if not result:
    print("⚠️ 未返回检测结果结构")
    sys.exit(0)

# 兼容多页结构
detections = result[0]
print(f"📦 检测框数量: {len(detections)}")

img = cv2.imread(str(IMG_PATH))
if img is None:
    print("❌ 无法读取图像")
    sys.exit(1)

h, w = img.shape[:2]
boxes_out = []
for i, det in enumerate(detections):
    if isinstance(det, list) or isinstance(det, tuple):
        box = det[0] if isinstance(det[0], (list, tuple, np.ndarray)) else det
    else:
        box = det
    box_np = np.array(box, dtype=np.float32)
    # 计算最小外接矩形
    minx = float(box_np[:,0].min()); maxx = float(box_np[:,0].max())
    miny = float(box_np[:,1].min()); maxy = float(box_np[:,1].max())
    boxes_out.append({
        'index': i,
        'quad': box_np.tolist(),
        'bbox': [int(minx), int(miny), int(maxx), int(maxy)],
        'width': int(maxx - minx + 1),
        'height': int(maxy - miny + 1)
    })
    cv2.polylines(img, [box_np.astype(np.int32)], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, str(i), (int(minx), int(miny)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)

out_dir = Path('output'); out_dir.mkdir(exist_ok=True)
json_path = out_dir/'ocr_det_only_boxes.json'
png_path = out_dir/'ocr_det_only_vis.png'
with open(json_path,'w',encoding='utf-8') as f:
    json.dump({'image': str(IMG_PATH), 'width': w, 'height': h, 'boxes': boxes_out}, f, ensure_ascii=False, indent=2)
cv2.imwrite(str(png_path), img)
print(f"🧾 已保存检测框 JSON: {json_path}")
print(f"🖼️ 已保存可视化: {png_path}")

# 粗略统计尺寸分布，辅助判断是否可能漏掉小后缀字符
small_boxes = [b for b in boxes_out if b['width']<=30 and b['height']<=30]
print(f"📊 总框 {len(boxes_out)} | 可能单字符小框 {len(small_boxes)}")
if small_boxes:
    print("   示例前5个小框:")
    for sb in small_boxes[:5]:
        print("    - idx={index} bbox={bbox} size={width}x{height}".format(**sb))
