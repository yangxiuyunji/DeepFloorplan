# -*- coding: utf-8 -*-
"""
参数对比探针：对 demo/demo2.jpg 进行 2 轮 OCR 识别，对比是否能出现漏掉的后缀字符 (卧室C / 卫B)。
轮次A: 原始尺寸，基线参数 (drop_score=0.30)
轮次B: 1.5x 放大 + 更低阈值 (drop_score=0.20, det_db_box_thresh=0.3)
输出:
  output/ocr_param_probe_result.json  两轮文本列表
  控制台打印差异
不修改主工程文件。
用法:
  python ocr_param_probe_demo2.py [可选: 图片路径]
"""
import sys, json, cv2, numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image

IMG_PATH = Path(sys.argv[1]) if len(sys.argv)>1 else Path('demo/demo2.jpg')
if not IMG_PATH.exists():
    print(f"❌ 输入图片不存在: {IMG_PATH}")
    sys.exit(1)

# ---------- 轮次A ----------
ocrA = PaddleOCR(lang='ch', det_db_thresh=0.2, det_db_box_thresh=0.4, det_db_unclip_ratio=2.5,
                 use_dilation=True, drop_score=0.3, ocr_version='PP-OCRv3')
resA = ocrA.ocr(str(IMG_PATH), cls=True)
textsA = []
for line in resA:
    for box, (txt, conf) in line:
        textsA.append({'text': txt, 'conf': conf})
print(f"🔎 轮次A 文本数: {len(textsA)}")

# ---------- 轮次B (放大 + 降阈值) ----------
img = Image.open(IMG_PATH).convert('RGB')
w,h = img.size
scale=1.5
img_big = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
TEMP_BIG='__temp_big_probe.png'
img_big.save(TEMP_BIG)
ocrB = PaddleOCR(lang='ch', det_db_thresh=0.18, det_db_box_thresh=0.3, det_db_unclip_ratio=2.3,
                 use_dilation=True, drop_score=0.20, ocr_version='PP-OCRv3')
resB = ocrB.ocr(TEMP_BIG, cls=True)
textsB = []
for line in resB:
    for box, (txt, conf) in line:
        textsB.append({'text': txt, 'conf': conf})
print(f"🔎 轮次B 文本数: {len(textsB)} (放大后)")

# 差异分析
setA = set(t['text'] for t in textsA)
setB = set(t['text'] for t in textsB)
new_in_B = sorted(setB - setA)
missing_in_B = sorted(setA - setB)

print("\n📊 新增文本(仅在轮次B出现):", new_in_B if new_in_B else '无')
print("📊 丢失文本(轮次A有B没有):", missing_in_B if missing_in_B else '无')

# 针对目标关键字的匹配
targets = ['卧室C','卫B','C','B']
for t in targets:
    hitA = any(t in x['text'] for x in textsA)
    hitB = any(t in x['text'] for x in textsB)
    print(f"   🔍 目标 '{t}': A={'Y' if hitA else 'N'} B={'Y' if hitB else 'N'}")

out = {
    'image': str(IMG_PATH),
    'roundA': textsA,
    'roundB': textsB,
    'new_in_B': new_in_B,
    'missing_in_B': missing_in_B
}
Path('output').mkdir(exist_ok=True)
with open('output/ocr_param_probe_result.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print("🧾 结果已写出: output/ocr_param_probe_result.json")

try:
    Path(TEMP_BIG).unlink()
except Exception:
    pass
