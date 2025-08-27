from utils.ocr_enhanced import extract_room_text, fuse_ocr_and_segmentation, text_to_label
import re

def _merge_room_suffixes(ocr_items, merge_distance=120):
    """最小策略: 将 '卧室' / '卫' 基础文本与附近单字母框合并成 卧室X / 卫X.
    不跨类型; 单字符必须是 A-Z 且置信度>=0.25. 生成的新 bbox 为外接矩形.
    原单字符条目被移除. 若基文本已自带后缀(A/B/C等)则跳过.
    """
    if not ocr_items:
        return ocr_items
    updated=[]; consumed=set()
    # 预抽取中心
    centers=[(i, it['bbox'][0]+it['bbox'][2]/2.0, it['bbox'][1]+it['bbox'][3]/2.0) for i,it in enumerate(ocr_items)]
    # 建立索引列表
    for i,it in enumerate(ocr_items):
        if i in consumed: continue
        text=it.get('text','').strip()
        # 检测是否为基础房间词
        base_type=None
        if re.fullmatch(r'卧室', text):
            base_type='卧室'
        elif re.fullmatch(r'卫', text):
            base_type='卫'
        # 若已带后缀(卧室A / 卫B 等)跳过合并
        if re.fullmatch(r'(卧室|卫)[A-Z]', text, re.IGNORECASE):
            updated.append(it); continue
        if base_type is None:
            updated.append(it); continue
        # 在其右侧或下方找一个最近单字母框
        bx,by=centers[i][1], centers[i][2]
        best_j=None; best_d=None
        for j,(idx,cx,cy) in enumerate(centers):
            if idx==i or idx in consumed: continue
            jt=ocr_items[idx].get('text','').strip()
            if not re.fullmatch(r'[A-Z]', jt, re.IGNORECASE):
                continue
            conf=ocr_items[idx].get('confidence',0)
            if conf<0.25: # 置信度最低约束
                continue
            dx=cx-bx; dy=cy-by
            # 要求至少有明显水平或垂直关系, 且总体距离限制
            if dx< -20: # 在左侧过多不合并
                continue
            dist=(dx*dx+dy*dy)**0.5
            if dist>merge_distance:
                continue
            # 偏向右侧或下方
            if best_d is None or dist<best_d:
                best_d=dist; best_j=idx
        if best_j is not None:
            # 合并
            base_bbox=it['bbox']; suf_bbox=ocr_items[best_j]['bbox']
            x1=min(base_bbox[0], suf_bbox[0]); y1=min(base_bbox[1], suf_bbox[1])
            x2=max(base_bbox[0]+base_bbox[2], suf_bbox[0]+suf_bbox[2])
            y2=max(base_bbox[1]+base_bbox[3], suf_bbox[1]+suf_bbox[3])
            new_item=it.copy()
            new_item['text']=f"{base_type}{ocr_items[best_j]['text'].upper()}"
            new_item['bbox']=(x1, y1, x2-x1, y2-y1)
            new_item['merged_suffix']=ocr_items[best_j]['text']
            new_item['merge_distance']=round(best_d,1)
            consumed.add(best_j)
            updated.append(new_item)
        else:
            updated.append(it)
    # 追加未被使用的其它项
    for k,it in enumerate(ocr_items):
        if k in consumed: continue
    return updated

class OCRRecognitionEngine:
    """第二层：OCR文字识别器 (拆分)."""
    def __init__(self):
        pass

    def recognize_text(self, original_img):
        print("🔍 [第2层-OCR识别器] 提取OCR文字信息...")
        ocr_items = extract_room_text(original_img)
        # 后缀合并前数量
        pre_n=len(ocr_items)
        ocr_items = _merge_room_suffixes(ocr_items)
        if len(ocr_items)!=pre_n:
            merged_cnt = pre_n - len([it for it in ocr_items if 'merged_suffix' not in it])
            print(f"🔗 [OCR后缀合并] 原始{pre_n}条 -> 合并后{len(ocr_items)}条 (合并{merged_cnt}个单字符后缀)")

        # ===== 双通道回退触发判定 =====
        # 若缺少可能的多实例后缀（例如存在 卧室A 卧室B 但不含 卧室C；存在 卫A 但不含 卫B），尝试二次宽松检测。
        texts_now = [it.get('text','') for it in ocr_items]
        need_second_pass = False
        def _has(pattern):
            import re; return any(re.fullmatch(pattern, t) for t in texts_now)
        # 卧室判定：有 卧室A 或 卧室B 但没有任何 卧室C-Z
        if any(t.startswith('卧室') for t in texts_now):
            # 如果出现过至少两个不同卧室后缀但没有 C，尝试补充
            import re
            suffixes = {m.group(1) for t in texts_now for m in [re.match(r'卧室([A-Z])$', t)] if m}
            if ('A' in suffixes or 'B' in suffixes) and 'C' not in suffixes:
                need_second_pass = True
        # 卫生间判定：有 卫A 但无 卫B
        if any(t.startswith('卫A') for t in texts_now) and not any(t.startswith('卫B') for t in texts_now):
            need_second_pass = True

        # 厨房缺失判定: 主通道没有任何含“厨/厨房/灶”文本 (防止被增强图像干扰漏检)
        kitchen_tokens = ('厨','厨房','灶')
        kitchen_missing = not any(any(k in t for k in kitchen_tokens) for t in texts_now)
        if kitchen_missing:
            print("🍳 [OCR厨房检测] 主通道未发现厨房相关文本，计划触发厨房专用回退扫描")

        if need_second_pass:
            print("🛰️ [OCR双通道] 触发第二通道宽松检测以查找缺失后缀 (卧室C/卫B)...")
            extra_items = self._second_pass_suffix_scan(original_img, existing=texts_now)
            if extra_items:
                print(f"🛰️ [OCR双通道] 第二通道新增 {len(extra_items)} 条候选 (含单字符/组合)")
                # 合并 + 去重（按文本 & IoU 简单过滤）
                ocr_items = self._merge_first_second_pass(ocr_items, extra_items)
                # 再做一次后缀合并
                after_second = _merge_room_suffixes(ocr_items)
                if len(after_second)!=len(ocr_items):
                    print("🔗 [OCR后缀合并] 第二通道后再次合并后缀")
                ocr_items = after_second
            else:
                print("🛰️ [OCR双通道] 第二通道未发现新增可用字符")

        # 厨房专用回退：放在后缀二次扫描之后，避免重复加载
        if kitchen_missing:
            kitchen_texts_now = [it.get('text','') for it in ocr_items]
            if not any(any(k in t for k in kitchen_tokens) for t in kitchen_texts_now):
                extra_k = self._second_pass_kitchen_scan(original_img, existing=kitchen_texts_now)
                if extra_k:
                    print(f"🍳 [OCR厨房检测] 回退扫描新增 {len(extra_k)} 条厨房候选")
                    ocr_items = self._merge_first_second_pass(ocr_items, extra_k)
                else:
                    print("🍳 [OCR厨房检测] 回退扫描仍未发现厨房文本")

        print(f"✅ [第2层-OCR识别器] 识别到 {len(ocr_items)} 条文字")
        # 兼容旧接口：返回 (items, 图像shape)
        return ocr_items, getattr(original_img, 'shape', None)

    def fuse_with_segmentation(self, floorplan, ocr_items):
        print("🔗 [第2层-OCR识别器] OCR 与语义分割结果融合...")
        fused = fuse_ocr_and_segmentation(floorplan, ocr_items)
        print("✅ [第2层-OCR识别器] 融合完成")
        return fused

    def map_text_to_label(self, text):
        return text_to_label(text)

    # ===== 第二通道宽松检测实现 =====
    def _second_pass_suffix_scan(self, original_img, existing):
        try:
            from paddleocr import PaddleOCR
            import numpy as np
            import cv2
        except Exception as e:
            print(f"⚠️ [OCR双通道] 无法导入PaddleOCR: {e}")
            return []
        # 减少增强副作用：直接用原图 (若是 PIL 转为 np)
        if hasattr(original_img, 'convert'):
            import numpy as _np
            img_np = _np.array(original_img.convert('RGB'))
        else:
            img_np = original_img
        ocr_fallback = PaddleOCR(
            lang='ch', det_db_thresh=0.18, det_db_box_thresh=0.35, det_db_unclip_ratio=2.3,
            drop_score=0.20, use_angle_cls=False, use_dilation=True, det_db_score_mode='fast'
        )
        try:
            results = ocr_fallback.ocr(img_np)
        except Exception as e:
            print(f"⚠️ [OCR双通道] 执行失败: {e}")
            return []
        new_items=[]
        def _norm_bbox(poly):
            xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
            x=min(xs); y=min(ys); w=max(xs)-x; h=max(ys)-y
            return x,y,w,h
        if results and isinstance(results, list):
            first = results[0]
            # 旧格式 list
            for line in first:
                if len(line)>=2:
                    poly=line[0]; txt=line[1][0]; conf=line[1][1]
                    if not txt.strip():
                        continue
                    # 保留：1) 与房间前缀相关 2) 单字符A-Z 3) 完整卧室X/卫X
                    keep=False
                    import re
                    if re.fullmatch(r'[A-Z]', txt):
                        if conf>=0.25: keep=True
                    if any(k in txt for k in ['卧室','卫']):
                        if conf>=0.25: keep=True
                    if not keep:
                        continue
                    if txt in existing:
                        # 若已有则跳过
                        continue
                    x,y,w,h = _norm_bbox(poly)
                    new_items.append({'text':txt,'bbox':(int(x),int(y),int(w),int(h)),'confidence':float(conf),'source':'second_pass'})
        return new_items

    def _merge_first_second_pass(self, first, second, iou_thr=0.5):
        # 简单 IoU 去重：若 second 与 first 任何 bbox IoU>thr 且文本相同则丢弃
        def _bbox(b):
            x,y,w,h=b; return (x,y,x+w,y+h)
        def _iou(a,b):
            ax1,ay1,ax2,ay2=_bbox(a); bx1,by1,bx2,by2=_bbox(b)
            inter_w=max(0,min(ax2,bx2)-max(ax1,bx1))
            inter_h=max(0,min(ay2,by2)-max(ay1,by1))
            inter=inter_w*inter_h
            if inter==0: return 0.0
            area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
            return inter/float(area_a+area_b-inter+1e-6)
        merged=list(first)
        for s in second:
            dup=False
            for f in first:
                if f.get('text')==s.get('text') and _iou(f['bbox'], s['bbox'])>iou_thr:
                    dup=True; break
            if not dup:
                merged.append(s)
        return merged

    # ===== 厨房专用第二通道 =====
    def _second_pass_kitchen_scan(self, original_img, existing):
        try:
            from paddleocr import PaddleOCR
            import numpy as np
        except Exception as e:
            print(f"⚠️ [OCR厨房回退] 无法导入PaddleOCR: {e}")
            return []
        # 使用原始彩色图（不做自定义增强）
        if hasattr(original_img, 'convert'):
            import numpy as _np
            img_np = _np.array(original_img.convert('RGB'))
        else:
            img_np = original_img if isinstance(original_img, np.ndarray) else np.array(original_img)
        try:
            ocr_k = PaddleOCR(
                lang='ch', det_db_thresh=0.19, det_db_box_thresh=0.38, det_db_unclip_ratio=2.6,
                drop_score=0.25, use_angle_cls=True, use_dilation=True, det_db_score_mode='fast'
            )
        except Exception as e:
            print(f"⚠️ [OCR厨房回退] 初始化失败: {e}")
            return []
        try:
            results = ocr_k.ocr(img_np)
        except Exception as e:
            print(f"⚠️ [OCR厨房回退] 执行失败: {e}")
            return []
        new_items=[]
        def _norm_bbox(poly):
            xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
            x=min(xs); y=min(ys); w=max(xs)-x; h=max(ys)-y
            return x,y,w,h
        kitchen_keys=('厨','厨房','灶')
        if results and isinstance(results, list) and len(results)>0:
            first = results[0]
            for line in first:
                if len(line)>=2:
                    poly=line[0]; txt=line[1][0]; conf=line[1][1]
                    if not txt.strip():
                        continue
                    if not any(k in txt for k in kitchen_keys):
                        continue
                    if txt in existing:
                        continue
                    if conf < 0.25:  # 低得分弃用
                        continue
                    x,y,w,h=_norm_bbox(poly)
                    new_items.append({'text':txt,'bbox':(int(x),int(y),int(w),int(h)),'confidence':float(conf),'source':'kitchen_pass'})
        return new_items
