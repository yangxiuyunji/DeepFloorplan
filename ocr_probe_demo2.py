import json, sys, cv2
from pathlib import Path
from PIL import Image
from engines.ocr_engine import OCRRecognitionEngine

def main():
    img_path = Path("demo/demo2.jpg")
    if not img_path.exists():
        print(f"❌ 找不到图片: {img_path}")
        return
    img = Image.open(img_path).convert("RGB")
    ocr = OCRRecognitionEngine()
    items, shape = ocr.recognize_text(img)  # 已在你的ocr_engine里返回 (list, shape)
    print(f"✅ OCR完成: 共 {len(items)} 条 (原图尺寸={shape})")
    texts = []
    for i, it in enumerate(items, 1):
        t = it.get("text","")
        conf = it.get("confidence",0)
        bbox = it.get("bbox")
        print(f"{i:02d}. text='{t}' conf={conf:.3f} bbox={bbox}")
        texts.append(t)

    # 关键目标检测
    target_words = ["卧室C","卫B","卧室","卫"]
    print("\n🔍 关键词匹配:")
    for tw in target_words:
        hits = [t for t in texts if tw in t]
        print(f" - {tw}: {'FOUND '+str(hits) if hits else '未找到'}")

    # 输出原始全部文本到 json 方便进一步人工核对
    out_json = Path("output/ocr_probe_demo2_raw.json")
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(items,f,ensure_ascii=False,indent=2)
    print(f"\n🧾 结果已写出: {out_json}")

if __name__ == "__main__":
    main()