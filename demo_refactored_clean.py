#!/usr/bin/env python3
"""
DeepFloorplan 房间检测 - 重构版本 (带坐标轴)
============================================

本文件是 demo.py 的完全重构版本，主要改进：
1. 面向对象设计替代过程式编程
2. 消除90%重复代码
3. 统一配置管理
4. 清晰的职责分离
5. 现代化代码风格
6. 坐标轴显示和房间坐标信息

功能完全等同于原版本，但代码更简洁、优雅、易维护。
"""

import os
import sys
import subprocess
import platform
import argparse
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime

# 配置环境
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

try:
    import tensorflow.compat.v1 as tf  # type: ignore
except Exception as _tf_err:
    class _DummyTF:
        def __getattr__(self, item):
            raise ImportError(f"TensorFlow 未安装，无法访问 {item}: {_tf_err}")
    tf = _DummyTF()  # fallback
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from PIL import ImageFont, ImageDraw, Image

# 引入四层架构组件
from engines.segmentation_engine import AISegmentationEngine
from engines.ocr_engine import OCRRecognitionEngine
from engines.fusion_engine import FusionDecisionEngine
from engines.post_rules import ReasonablenessValidator

## 原第一至三层及第四层规则类已拆分到 engines/ 下, 此处不再定义重复实现，以下开始主处理器类

# ================= 中文字体配置（防止 Matplotlib / OCR 子图出现问号）=================
# 说明: 之前的自动检测只在 matplotlib 目录下找字体, 实际 Windows 中文字体在 C:/Windows/Fonts 下, 导致未找到 -> 问号。
# 策略: 1) 优先系统常见字体 2) 其次项目自带 fonts/ 3) 最后退回默认字体 (仍可显示英文, 但提示).

def _find_chinese_font():
    candidates = [
        # Windows 常见字体
        r"C:/Windows/Fonts/msyh.ttc",
        r"C:/Windows/Fonts/msyh.ttf",
        r"C:/Windows/Fonts/msyhl.ttc",
        r"C:/Windows/Fonts/simhei.ttf",
        r"C:/Windows/Fonts/simhei.ttc",
        r"C:/Windows/Fonts/simsun.ttc",
        r"C:/Windows/Fonts/simfang.ttf",
        r"C:/Windows/Fonts/STSONG.TTF",
        # 项目内自带 (可自行添加)
        str(Path(__file__).parent / "fonts" / "msyh.ttc"),
        str(Path(__file__).parent / "fonts" / "simhei.ttf"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

CH_FONT_PATH = _find_chinese_font()
if CH_FONT_PATH:
    try:
        # 显式注册字体，避免仅使用 stem 导致找不到 family 名称
        try:
            from matplotlib import font_manager as _fm
            _fm.fontManager.addfont(CH_FONT_PATH)
        except Exception:
            pass
        CH_FONT = FontProperties(fname=CH_FONT_PATH)
        # 常见中文字体别名，提升匹配成功率
        matplotlib.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'MS YaHei', '微软雅黑', 'SimHei', 'SimSun', 'Heiti SC', 'Noto Sans CJK SC'
        ]
        # 追加当前字体文件对应的名称（可能是 msyh / simhei 等）
        stem_name = Path(CH_FONT_PATH).stem
        if stem_name not in matplotlib.rcParams['font.sans-serif']:
            matplotlib.rcParams['font.sans-serif'].append(stem_name)
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"🈶 已加载中文字体: {CH_FONT_PATH}")
    except Exception as _fe:
        print(f"⚠️ 中文字体加载失败, 使用默认字体: {_fe}")
        CH_FONT = FontProperties()
else:
    print("⚠️ 未找到可用中文字体, 可能出现问号。可将 ms yh / simhei 字体放入 fonts/ 目录。")
    CH_FONT = FontProperties()

class SizeConstraintEngine:  # 占位避免旧引用; 实际逻辑已在 engines.post_rules 中
    pass

class BuildingBoundaryDetector:  # 占位避免旧引用; 实际逻辑已在 engines.post_rules 中
    pass


class FloorplanProcessor:
    """户型图处理器 - 四层智能决策架构统一管理器"""

    def __init__(self, model_path="pretrained"):
        """初始化四层架构处理器"""
        if getattr(tf, "__class__", type(tf)).__name__ == "_DummyTF":
            raise ImportError("请安装 TensorFlow ≥ 1.x")

        print("🏠 DeepFloorplan 房间检测 - 四层智能决策架构")
        print("=" * 60)

        # 初始化四层架构组件
        self.ai_engine = AISegmentationEngine(model_path)
        self.ocr_engine = OCRRecognitionEngine()
        self.fusion_engine = FusionDecisionEngine()
        self.validator = ReasonablenessValidator()

        # 运行时缓存/状态
        self.last_enhanced = None  # 最近一次增强后的 label 图 (512x512)
        self._boundary_cache = {}  # {md5: 增强后含墙体结果}

    def load_model(self):
        """加载AI分割模型"""
        if getattr(tf, "__class__", type(tf)).__name__ == "_DummyTF":
            raise ImportError("请安装 TensorFlow ≥ 1.x")
        self.ai_engine.load_model()

    def preprocess_image(self, image_path):
        """图像预处理"""
        print(f"📸 处理图像: {image_path}")

        # 读取图像
        img = Image.open(image_path).convert("RGB")
        original_size = img.size

        print(f"📏 原始图像尺寸: {original_size[0]} x {original_size[1]} (宽x高)")

        # 调整到模型输入尺寸 (512x512)
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        print(f"🔄 神经网络输入: 512 x 512 (固定尺寸)")

        return img_array, original_size, np.array(img)

    def process_with_four_layer_architecture(self, img_array, original_img, original_size):
        """使用四层架构处理图像"""
        print("\n🏗️ 开始四层智能决策处理流程...")
        
        # 第一层：AI语义分割
        ai_prediction = self.ai_engine.segment_image(img_array)
        
        # 第二层：OCR文字识别
        ocr_results, ocr_shape = self.ocr_engine.recognize_text(original_img)
        
        # 第三层：融合决策
        fused_results = self.fusion_engine.fuse_results(ai_prediction, ocr_results, ocr_shape)
        
        # 第四层：合理性验证
        validated_results = self.validator.validate_and_correct(fused_results, ocr_results, original_size)
        
        # 保存结果用于统计
        self.last_enhanced = validated_results
        
        print("🎉 四层智能决策处理完成！")
        return {
            'ai_raw': ai_prediction,
            'ocr_results': ocr_results, 
            'fusion_result': fused_results,
            'final_result': validated_results
        }

    # 保留原有接口以保持兼容性
    def run_inference(self, img_array):
        """运行神经网络推理（兼容接口）"""
        return self.ai_engine.segment_image(img_array)

    def extract_ocr_info(self, original_img):
        """提取OCR文字信息（兼容接口）"""
        return self.ocr_engine.recognize_text(original_img)

    def fuse_predictions(self, prediction, room_text_items, ocr_shape):
        """融合预测结果（兼容接口）"""
        return self.fusion_engine.fuse_results(prediction, room_text_items, ocr_shape)

    def detect_rooms(self, enhanced, room_text_items, original_size):
        """检测各类房间（兼容接口）"""
        # 现在这个功能已经整合到四层架构中
        return enhanced
        
    def _clean_misidentified_regions(self, enhanced, room_text_items, original_size):
        """清理AI分割中的误识别区域，只保留OCR验证的房间区域"""
        print("🧹 清理AI分割误识别区域...")
        
        # 获取OCR验证的房间位置
        ocr_rooms = {}
        for item in room_text_items:
            text = item["text"].lower().strip()
            room_type = None
            
            if any(keyword in text for keyword in ["厨房", "kitchen", "厨"]):
                room_type = 7  # 厨房
            elif any(keyword in text for keyword in ["卫生间", "bathroom", "卫", "洗手间", "浴室", "淋浴间", "shower", "淋浴", "盥洗室"]):
                room_type = 2  # 卫生间
            elif any(keyword in text for keyword in ["客厅", "living", "厅", "起居室"]):
                room_type = 3  # 客厅
            elif any(keyword in text for keyword in ["卧室", "bedroom", "主卧", "次卧"]):
                room_type = 4  # 卧室
            elif any(keyword in text for keyword in ["阳台", "balcony", "阳兮", "阳合", "阳囊"]):
                room_type = 6  # 阳台
            elif any(keyword in text for keyword in ["书房", "study", "办公室", "office"]):
                room_type = 8  # 书房
                
            if room_type:
                if room_type not in ocr_rooms:
                    ocr_rooms[room_type] = []
                
                # 转换OCR坐标到512x512坐标系
                x, y, w, h = item["bbox"]
                # OCR是在2倍放大图像上，需要转换到512x512
                ocr_to_512_x = 512.0 / (original_size[0] * 2)
                ocr_to_512_y = 512.0 / (original_size[1] * 2) 
                
                center_512_x = int((x + w//2) * ocr_to_512_x)
                center_512_y = int((y + h//2) * ocr_to_512_y)
                ocr_rooms[room_type].append((center_512_x, center_512_y, item["confidence"]))
        
        # 对于每个房间类型，只保留OCR验证位置附近的分割区域
        for room_label, room_positions in ocr_rooms.items():
            if room_label in [2, 7]:  # 处理卫生间和厨房的误识别问题
                room_name = "卫生间" if room_label == 2 else "厨房"
                print(f"🧹 清理{room_name}误识别区域，保留{len(room_positions)}个OCR验证位置")
                
                # 获取所有指定标签的像素
                mask = (enhanced == room_label).astype(np.uint8)
                
                # 使用连通域分析找到所有区域
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
                
                # 创建新的清理后的mask
                cleaned_mask = np.zeros_like(mask)
                
                for comp_id in range(1, num_labels):  # 跳过背景(0)
                    comp_centroid = centroids[comp_id]
                    comp_center_x, comp_center_y = int(comp_centroid[0]), int(comp_centroid[1])
                    comp_area = stats[comp_id, cv2.CC_STAT_AREA]
                    
                    # 检查这个连通域是否接近任何OCR验证的位置
                    is_valid = False
                    min_distance = float('inf')
                    closest_confidence = 0
                    
                    for ocr_x, ocr_y, confidence in room_positions:
                        distance = np.sqrt((comp_center_x - ocr_x)**2 + (comp_center_y - ocr_y)**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_confidence = confidence
                    
                    # 设置距离阈值：根据OCR验证位置数量动态调整
                    if len(room_positions) > 1:
                        # 多个OCR位置时，使用更宽松的阈值
                        distance_threshold = 120 if room_label == 7 else 100
                    else:
                        # 单个OCR位置时，使用标准阈值  
                        distance_threshold = 100 if room_label == 7 else 80
                    
                    # 设置面积阈值：防止巨大的误识别区域
                    max_area_threshold = 15000 if room_label == 7 else 10000  # 适当放宽面积限制
                    
                    # 如果距离在合理范围内且面积不超标，认为是有效的
                    if min_distance < distance_threshold and comp_area < max_area_threshold:  
                        is_valid = True
                        print(f"   ✅ 保留{room_name}区域：中心({comp_center_x}, {comp_center_y}), 距OCR: {min_distance:.1f}像素, 面积: {comp_area}, 置信度: {closest_confidence:.3f}")
                    else:
                        reasons = []
                        if min_distance >= distance_threshold:
                            reasons.append("距离过远")
                        if comp_area >= max_area_threshold:
                            reasons.append("面积过大")
                        print(f"   ❌ 移除误识别区域：中心({comp_center_x}, {comp_center_y}), 距OCR: {min_distance:.1f}像素, 面积: {comp_area} ({', '.join(reasons)})")
                    
                    if is_valid:
                        # 保留这个连通域
                        component_mask = (labels_im == comp_id)
                        cleaned_mask[component_mask] = 1
                
                # 用清理后的mask更新enhanced
                enhanced[mask == 1] = 0  # 先清除所有原来的标记
                enhanced[cleaned_mask == 1] = room_label  # 然后设置验证过的区域
                
                removed_pixels = np.sum(mask) - np.sum(cleaned_mask)
                print(f"   📊 清理结果：移除了 {removed_pixels} 个误识别像素")
        
        return enhanced

    def _apply_color_mapping(self, result_array, original_size):
        """将分割结果应用颜色映射"""
        # 调整到原始尺寸
        result_resized = cv2.resize(result_array, original_size, interpolation=cv2.INTER_NEAREST)

        # 使用缓存的边界增强，避免重复多次细化破坏结构
        result_with_boundaries = self._add_boundary_detection_cached(result_resized)

        # 生成彩色图
        h, w = result_with_boundaries.shape
        colored_result = np.zeros((h, w, 3), dtype=np.uint8)
        # 与图例统一: 使用 floorplan_fuse_map_figure
        from utils.rgb_ind_convertor import floorplan_fuse_map_figure as _COLOR_MAP
        for label_value, color in _COLOR_MAP.items():
            if label_value > 10:  # 安全过滤（现用到0-10）
                continue
            mask = (result_with_boundaries == label_value)
            colored_result[mask] = color

        return colored_result

    def _visualize_ocr_results(self, original_img, room_text_items):
        """可视化OCR识别结果（显示修正后的文本）"""
        ocr_img = original_img.copy()

        # 定义房间类型颜色
        room_colors = {
            "厨房": (0, 255, 0),      # 绿色
            "卫生间": (255, 0, 0),    # 蓝色
            "客厅": (0, 165, 255),    # 橙色
            "卧室": (128, 0, 128),    # 紫色
            "阳台": (255, 255, 0),    # 青色
            "书房": (165, 42, 42),    # 棕色
        }

        # OCR修正映射 - 修正常见的OCR识别错误
        ocr_corrections = {
            # 阳台相关修正
            "阳兮": "阳台",
            "阳台": "阳台",
            "陽台": "阳台",
            "阳合": "阳台",
            "阳舍": "阳台",
            "阳古": "阳台",

            # 厨房相关修正
            "厨房": "厨房",
            "廚房": "厨房",
            "厨户": "厨房",
            "厨庐": "厨房",
            "庁房": "厨房",

            # 卫生间相关修正
            "卫生间": "卫生间",
            "衛生間": "卫生间",
            "卫生闬": "卫生间",
            "卫生门": "卫生间",
            "浴室": "卫生间",
            "洗手间": "卫生间",
            "厕所": "卫生间",

            # 客厅相关修正
            "客厅": "客厅",
            "客廳": "客厅",
            "客应": "客厅",
            "客广": "客厅",
            "起居室": "客厅",
            "会客厅": "客厅",

            # 卧室相关修正
            "卧室": "卧室",
            "臥室": "卧室",
            "卧宝": "卧室",
            "卧窒": "卧室",
            "卧空": "卧室",
            "网房": "卧室",
            "主卧": "主卧",
            "次卧": "次卧",

            # 书房相关修正
            "书房": "书房",
            "書房": "书房",
            "书户": "书房",
            "书庐": "书房",
            "学习室": "书房",
            "工作室": "书房",

            # 餐厅相关修正
            "餐厅": "餐厅",
            "餐廳": "餐厅",
            "饭厅": "餐厅",
            "用餐区": "餐厅",

            # 入户相关修正
            "入户": "入户",
            "玄关": "入户",
            "门厅": "入户",

            # 走廊相关修正
            "走廊": "走廊",
            "过道": "走廊",
            "通道": "走廊",

            # 储物相关修正
            "储物间": "储物间",
            "储藏室": "储物间",
            "杂物间": "储物间",
            "衣帽间": "衣帽间",

            # 清理单字符噪音（常见OCR错误识别）
            "门": "",
            "户": "",
            "口": "",
            "人": "",
            "大": "",
            "小": "",
            "中": "",
            "上": "",
            "下": "",
            "左": "",
            "右": "",
            "一": "",
            "二": "",
            "三": "",
            "四": "",
            "五": "",
            "1": "",
            "2": "",
            "3": "",
            "4": "",
            "5": "",
            "6": "",
            "7": "",
            "8": "",
            "9": "",
            "0": "",
            "m": "",
            "M": "",
            "㎡": "",
            "平": "",
            "方": "",
            "米": "",
        }

        for item in room_text_items:
            original_text = item["text"]
            bbox = item["bbox"]
            confidence = item.get("confidence", 1.0)

            # ===== 坐标缩放修正 =====
            # OCR阶段图像被放大2倍(或其他比例)，当前bbox仍处于OCR坐标系，需要映射回原图
            ocr_w = item.get('ocr_width')
            ocr_h = item.get('ocr_height')
            if ocr_w and ocr_h:
                scale_x = ocr_w / original_img.shape[1]
                scale_y = ocr_h / original_img.shape[0]
            else:
                # 回退：如果坐标明显超出原图尺寸，假设放大2倍
                scale_x = scale_y = 2.0 if (bbox[0] > original_img.shape[1] or bbox[1] > original_img.shape[0]) else 1.0

            x, y, w, h = bbox
            if scale_x != 1.0 or scale_y != 1.0:
                x = int(x / scale_x)
                y = int(y / scale_y)
                w = max(1, int(w / scale_x))
                h = max(1, int(h / scale_y))
            # 使用缩放后的局部变量，不修改原始数据

            # 1. 跳过空文本或纯空白
            if not original_text or not original_text.strip():
                continue

            # 2. 跳过低置信度的单字符（这些通常是噪音）
            if len(original_text.strip()) == 1 and confidence < 0.8:
                continue

            # 3. 跳过纯数字、纯符号文本
            cleaned_for_check = original_text.strip()
            if cleaned_for_check.isdigit() or not any(c.isalpha() or c in '厨房客厅卧室卫生间阳台书餐储衣玄走廊过道入户' for c in cleaned_for_check):
                continue

            # 4. 跳过长度过短且不包含房间关键词的文本
            if len(cleaned_for_check) < 2 and not any(keyword in cleaned_for_check for keyword in ['厨', '卫', '客', '卧', '阳', '书', '餐']):
                continue

            # 5. 跳过包含明显乱码字符的文本
            garbage_chars = {'�', '□', '■', '▲', '▼', '◆', '●', '○', '※', '★', '☆'}
            if any(char in original_text for char in garbage_chars):
                continue

            # 6. 应用OCR修正
            display_text = ocr_corrections.get(original_text, original_text)

            # 7. 如果修正后为空，则跳过
            if not display_text or not display_text.strip():
                continue

            # 8. 最后检查：如果修正后仍然是单字符且置信度不高，跳过
            if len(display_text.strip()) == 1 and confidence < 0.9:
                continue

            # 确定房间类型和颜色
            color = (255, 255, 255)  # 默认白色
            for room_type, room_color in room_colors.items():
                if any(keyword in display_text.lower() for keyword in [room_type.lower(),
                      {"厨房": "kitchen", "卫生间": "bathroom", "客厅": "living",
                       "卧室": "bedroom", "阳台": "balcony", "书房": "study"}.get(room_type, "")]):
                    color = room_color
                    break

            # 绘制文字边界框
            cv2.rectangle(ocr_img, (x, y), (x + w, y + h), color, 2)

            # 标签文本处理
            if original_text != display_text and display_text:
                label = f"{display_text} (修正:{original_text})"
                label_color = (255, 165, 0)
            elif display_text:
                label = f"{display_text}"
                label_color = color
            else:
                continue

            font_scale = max(0.5, min(1.0, w / 100))
            thickness = max(1, int(font_scale * 2))

            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in label)
            if has_chinese and CH_FONT_PATH:
                # PIL 路径：只描边 + 半透明背景（避免纯白块）
                font_size = max(12, min(32, int(w * 0.5)))
                try:
                    pil_font = ImageFont.truetype(CH_FONT_PATH, font_size)
                except Exception:
                    pil_font = ImageFont.load_default()
                # 转 RGBA 以实现半透明
                pil_img = Image.fromarray(ocr_img)
                if pil_img.mode != 'RGBA':
                    pil_img = pil_img.convert('RGBA')
                draw = ImageDraw.Draw(pil_img, 'RGBA')
                text_bbox = draw.textbbox((0, 0), label, font=pil_font)
                tw = text_bbox[2] - text_bbox[0]
                th = text_bbox[3] - text_bbox[1]
                place_above = y - th - 6 >= 0
                if place_above:
                    text_x = x
                    text_y = y - th - 4
                else:
                    text_x = x
                    text_y = y + 2
                bg_x1 = text_x - 3
                bg_y1 = text_y - 2
                bg_x2 = text_x + tw + 3
                bg_y2 = text_y + th + 2
                # 半透明背景 (白 30% 透明)
                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 70), outline=label_color + (255,), width=1)
                draw.text((text_x, text_y), label, font=pil_font, fill=label_color + (255,))
                # 回到BGR
                ocr_img = np.array(pil_img.convert('RGB'))
            else:
                # OpenCV 路径：用 alpha 混合生成半透明背景
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tw, th = text_size
                place_above = y - th - 6 >= 0
                if place_above:
                    text_x = x
                    text_y = y - 4
                    box_y1 = y - th - 6
                    box_y2 = y - 2
                else:
                    text_x = x
                    text_y = y + th + 2
                    box_y1 = y
                    box_y2 = y + th + 8
                box_x1 = x - 2
                box_x2 = x + tw + 4
                # 边界裁剪
                h_img, w_img = ocr_img.shape[:2]
                box_x1_c = max(0, box_x1); box_y1_c = max(0, box_y1)
                box_x2_c = min(w_img - 1, box_x2); box_y2_c = min(h_img - 1, box_y2)
                if box_x2_c > box_x1_c and box_y2_c > box_y1_c:
                    roi = ocr_img[box_y1_c:box_y2_c, box_x1_c:box_x2_c]
                    overlay = roi.copy()
                    overlay[:] = (255, 255, 255)
                    cv2.addWeighted(overlay, 0.3, roi, 0.7, 0, roi)
                    ocr_img[box_y1_c:box_y2_c, box_x1_c:box_x2_c] = roi
                cv2.rectangle(ocr_img, (box_x1, box_y1), (box_x2, box_y2), label_color, 1)
                cv2.putText(ocr_img, label, (text_x, text_y - 6 if place_above else text_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, label_color, thickness, cv2.LINE_AA)

        return ocr_img

    def _add_room_annotations(self, ax, room_info):
        """在图上添加房间标注"""
        for room_type, room_list in room_info.items():
            for i, coords in enumerate(room_list):
                if coords["pixels"] > 0:
                    center_x, center_y = coords["center"]
                    bbox = coords["bbox"]
                    raw_text = coords.get("text", "")
                    is_fallback = (raw_text == "分割检测") or coords.get('source') == 'segmentation_fallback'

                    # 标注房间中心点
                    ax.plot(center_x, center_y, "o", markersize=10, color="white",
                           markeredgecolor="black", markeredgewidth=2)

                    # 房间标注
                    display_name = room_type
                    # 若存在原始OCR文本且不是分割回退，优先显示原文本（保留 A/B/C 等后缀）
                    if raw_text and not is_fallback and raw_text != room_type:
                        display_name = raw_text
                    # 多实例加序号（同时仍保留具体文本）
                    if len(room_list) > 1 and not raw_text.startswith(display_name):
                        display_name = f"{display_name}#{i+1}"
                    label_text = f"{display_name}\n({center_x},{center_y})"

                    ax.annotate(label_text, xy=(center_x, center_y), xytext=(10, 10),
                                textcoords="offset points", fontsize=10, fontweight="bold",
                                fontproperties=CH_FONT,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))

                    # 绘制边界框
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="red", linewidth=2, linestyle="--")
                    ax.add_patch(rect)

    def _add_color_legend(self, fig):
        """添加颜色图例（与实际着色严格一致）

        之前版本图例颜色是手写/示意色，导致与实际 floorplan_fuse_map_figure 中的颜色不一致。
        这里直接读取 utils.rgb_ind_convertor.floorplan_fuse_map_figure，确保完全同步。
        """
        from utils.rgb_ind_convertor import floorplan_fuse_map_figure as _COLOR_MAP

        # 标签 -> 中文名称映射（只展示主要房型 + 墙体/开口）
        label_name_map = {
            7: "厨房",
            2: "卫生间",
            3: "客厅",
            4: "卧室",
            6: "阳台",
            8: "书房",
            9: "开口",
            10: "墙体",
        }

        legend_elements = []
        ordered_labels = [7,2,3,4,6,8,9,10]
        print("🎨 图例同步检查: ")
        for label_id in ordered_labels:  # 固定排序，方便阅读
            if label_id not in _COLOR_MAP:
                continue
            name = label_name_map.get(label_id, str(label_id))
            rgb = _COLOR_MAP[label_id]
            print(f"   - {name} (label {label_id}) 颜色 RGB={rgb}")
            # 转 0-1 范围 (matplotlib 需要 RGB 顺序；原映射就是 RGB)
            color = np.array(rgb) / 255.0
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name))

        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_elements),
            fontsize=12,
            prop=CH_FONT,
            frameon=True,
        )

    def generate_results(self, ai_raw_result, ocr_result, fusion_result, final_result,
                         original_img, original_size, output_path, room_text_items):
        """生成四宫格对比结果图像（修正中文字体显示为 ? 的问题）。"""
        print("🎨 生成四宫格对比结果图像...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        room_info = self._extract_room_coordinates(final_result, original_size, room_text_items)
        self.last_room_info = room_info

        # AI 分割
        ai_colored = self._apply_color_mapping(ai_raw_result, original_size)
        ax1.imshow(cv2.addWeighted(original_img, 0.5, ai_colored, 0.5, 0))
        ax1.set_title("🤖 AI语义分割结果", fontsize=14, fontweight="bold", fontproperties=CH_FONT)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("X坐标 (像素)", fontsize=12, fontproperties=CH_FONT)
        ax1.set_ylabel("Y坐标 (像素)", fontsize=12, fontproperties=CH_FONT)

        # OCR
        ocr_colored = self._visualize_ocr_results(original_img, room_text_items)
        ax2.imshow(ocr_colored)
        ax2.set_title("🔍 OCR文字识别结果", fontsize=14, fontweight="bold", fontproperties=CH_FONT)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("X坐标 (像素)", fontsize=12, fontproperties=CH_FONT)
        ax2.set_ylabel("Y坐标 (像素)", fontsize=12, fontproperties=CH_FONT)

        # 融合
        fusion_colored = self._apply_color_mapping(fusion_result, original_size)
        ax3.imshow(cv2.addWeighted(original_img, 0.5, fusion_colored, 0.5, 0))
        ax3.set_title("🔗 AI+OCR融合结果", fontsize=14, fontweight="bold", fontproperties=CH_FONT)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("X坐标 (像素)", fontsize=12, fontproperties=CH_FONT)
        ax3.set_ylabel("Y坐标 (像素)", fontsize=12, fontproperties=CH_FONT)

        # 最终
        final_colored = self._apply_color_mapping(final_result, original_size)
        final_overlay = cv2.addWeighted(original_img, 0.5, final_colored, 0.5, 0)

        # ===== 竖线(x=600)调试辅助 =====
        try:
            ow, oh = original_size
            probe_x = 600
            if ow > probe_x and final_result is not None:
                # final_result 仍在 512 尺度 (宽=512) => 映射列索引
                fr_w = final_result.shape[1]
                mapped_col = int(round(probe_x * fr_w / float(ow)))
                col_vals = final_result[:, mapped_col]
                import numpy as _np
                uniq, cnt = _np.unique(col_vals, return_counts=True)
                dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
                wall_len = dist.get(10, 0) + dist.get(9, 0)
                continuous_wall = wall_len >= (final_result.shape[0] * 0.95)
                print(f"🔍 [竖线诊断] 原图x={probe_x} -> 512列={mapped_col}, 标签分布={dist}, 是否几乎整列墙体={continuous_wall}")
                if not continuous_wall:
                    print("✅ 判定: 该竖线更可能是可视化网格/叠加伪影, 不影响识别逻辑")
                else:
                    print("⚠️ 判定: 该列接近整列墙体, 可能来源于墙体细化算法, 可进一步排查 _add_boundary_detection 中 endpoint 连接逻辑")
        except Exception as _e:
            print(f"⚠️ [竖线诊断] 发生异常: {_e}")
        ax4.imshow(final_overlay)
        ax4.set_title("✅ 合理性验证后最终结果", fontsize=14, fontweight="bold", fontproperties=CH_FONT)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel("X坐标 (像素)", fontsize=12, fontproperties=CH_FONT)
        ax4.set_ylabel("Y坐标 (像素)", fontsize=12, fontproperties=CH_FONT)

        self._add_room_annotations(ax4, room_info)
        self._add_color_legend(fig)
        plt.tight_layout()

        os.makedirs("output", exist_ok=True)
        comparison_output = f"output/{output_path}_coordinate_result.png"
        plt.savefig(comparison_output, dpi=300, bbox_inches="tight")
        print(f"📊 四宫格对比结果已保存: {comparison_output}")

        # 单独最终结果
        plt.figure(figsize=(12, 8))
        plt.imshow(final_overlay)
        plt.title("房间检测最终结果", fontsize=16, fontweight="bold", fontproperties=CH_FONT)
        plt.grid(True, alpha=0.3)
        plt.xlabel("X坐标 (像素)", fontsize=12, fontproperties=CH_FONT)
        plt.ylabel("Y坐标 (像素)", fontsize=12, fontproperties=CH_FONT)
        for room_type, room_list in room_info.items():
            for i, coords in enumerate(room_list):
                if coords["pixels"] <= 0:
                    continue
                center_x, center_y = coords["center"]
                x1, y1, x2, y2 = coords["bbox"]
                plt.plot(center_x, center_y, "o", markersize=10, color="white",
                         markeredgecolor="black", markeredgewidth=2)
                if len(room_list) > 1:
                    label_text = f"{room_type}{i+1}\n({center_x},{center_y})"
                else:
                    label_text = f"{room_type}\n({center_x},{center_y})"
                plt.annotate(label_text, xy=(center_x, center_y), xytext=(10, 10),
                             textcoords="offset points", fontsize=10, fontweight="bold",
                             fontproperties=CH_FONT,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                     edgecolor="red", linewidth=2, linestyle="--")
                plt.gca().add_patch(rect)
        standard_output = f"output/{output_path}_result.png"
        plt.savefig(standard_output, dpi=300, bbox_inches="tight")
        print(f"📸 标准结果已保存: {standard_output}")
        plt.close('all')

        # ====== 导出边界文件（单独）======
        try:
            boundary_labeled = self._add_boundary_detection_cached(final_result.copy())  # 在 512x512 空间
            # 提取仅包含 9/10 标签的边界掩膜
            boundary_only = np.zeros_like(boundary_labeled)
            boundary_only[np.isin(boundary_labeled, [9, 10])] = boundary_labeled[np.isin(boundary_labeled, [9, 10])]

            # 生成彩色边界图 (白底 + 开口洋红 + 墙体黑)
            boundary_color = np.full((boundary_only.shape[0], boundary_only.shape[1], 3), 255, dtype=np.uint8)
            boundary_color[boundary_only == 9] = [255, 60, 128]   # openings
            boundary_color[boundary_only == 10] = [0, 0, 0]       # walls

            # 放大回原始尺寸
            boundary_color_resized = cv2.resize(boundary_color, original_size, interpolation=cv2.INTER_NEAREST)
            boundary_mask_binary = np.zeros((boundary_only.shape[0], boundary_only.shape[1]), dtype=np.uint8)
            boundary_mask_binary[boundary_only == 10] = 255  # 墙体
            boundary_mask_binary[boundary_only == 9] = 128   # 开口
            boundary_mask_resized = cv2.resize(boundary_mask_binary, original_size, interpolation=cv2.INTER_NEAREST)

            boundary_png = f"output/{output_path}_boundary.png"
            boundary_mask_png = f"output/{output_path}_boundary_mask.png"
            cv2.imwrite(boundary_png, cv2.cvtColor(boundary_color_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(boundary_mask_png, boundary_mask_resized)
            print(f"🧱 边界彩色图已保存: {boundary_png}")
            print(f"🧱 边界掩膜图已保存: {boundary_mask_png}")

            # 追加：墙体 + 开口 矢量化 & SVG/DXF 导出
            try:
                vec_data = self._vectorize_walls(boundary_mask_binary, original_size)
                svg_out = f"output/{output_path}_walls.svg"
                dxf_out = f"output/{output_path}_walls.dxf"
                self._export_walls_svg(svg_out, vec_data, original_size)
                self._export_walls_dxf(dxf_out, vec_data, original_size)
                print(f"🗺️ 矢量墙体导出: SVG {len(vec_data['walls_segments'])} 段 / DXF 多段线 {len(vec_data['walls_polylines'])}")
            except Exception as _ve:
                print(f"⚠️ 矢量墙体导出失败: {_ve}")
        except Exception as e:
            print(f"⚠️ 边界导出失败: {e}")

        # 结构化JSON导出
        try:
            json_path = f"output/{output_path}_result.json"
            self._export_room_json(room_info, original_size, json_path, room_text_items, image_output=standard_output)
            print(f"🧾 结构化房间数据已保存: {json_path}")
        except Exception as e:
            print(f"⚠️ JSON导出失败: {e}")
        return standard_output

    def _export_room_json(self, room_info, original_size, json_path, ocr_items, image_output=None):
        """导出房间识别结果为JSON，便于后续风水/空间分析。

        JSON结构:
        {
          "meta": { 原图尺寸/时间/输出图像等 },
          "rooms": [
             {
               "type": "卧室",
               "index": 1,              # 同类型序号（从1起）
               "label_id": 4,
               "center": {"x":123, "y":245},
               "center_normalized": {"x":0.35, "y":0.62},
               "bbox": {"x1":..,"y1":..,"x2":..,"y2":..,"width":..,"height":..},
               "area_pixels": 3456,      # 目前基于bbox像素估计（若需真实mask面积可后续扩展）
               "text_raw": "卧室",
               "confidence": 0.91,
               "distance_to_center": 210.4,
               "direction_8": "东北"     # 以图像上方为北，左为西
             }, ...
          ]
        }
        """
        orig_w, orig_h = original_size
        img_cx, img_cy = orig_w / 2.0, orig_h / 2.0

        # 房间中文到label映射（与 _extract_room_coordinates 中一致）
        name_to_label = {"厨房":7, "卫生间":2, "客厅":3, "卧室":4, "阳台":6, "书房":8}

        def direction_from_vector(dx, dy):
            # 图像坐标: y向下 -> 北在上方 => dy<0 为北
            angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360  # 0=东, 90=北
            dirs = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
            idx = int(((angle + 22.5) % 360) / 45)
            return dirs[idx]

        rooms_json = []
        for room_type, room_list in room_info.items():
            for idx, info in enumerate(room_list, start=1):
                if info.get('pixels', 0) <= 0:
                    continue
                cx, cy = info['center']
                x1, y1, x2, y2 = info['bbox']
                width = info.get('width', x2 - x1 + 1)
                height = info.get('height', y2 - y1 + 1)
                # 使用bbox面积作为近似
                area_pixels = width * height
                dx = cx - img_cx
                dy = cy - img_cy
                dist = float(np.hypot(dx, dy))
                direction = direction_from_vector(dx, dy)
                rooms_json.append({
                    "type": room_type,
                    "index": idx,
                    "label_id": name_to_label.get(room_type, -1),
                    "center": {"x": int(cx), "y": int(cy)},
                    "center_normalized": {"x": round(cx / orig_w, 4), "y": round(cy / orig_h, 4)},
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "width": int(width), "height": int(height)},
                    "area_pixels": int(area_pixels),
                    "text_raw": info.get('text', ''),
                    "confidence": round(float(info.get('confidence', 0.0)), 4),
                    "distance_to_center": round(dist, 2),
                    "direction_8": direction,
                })

        data = {
            "meta": {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "image_width": orig_w,
                "image_height": orig_h,
                "rooms_detected": len(rooms_json),
                "output_image": image_output,
                "note": "方向基于图像上北下南左西右东的默认假设; 若图纸朝向不同需调整。"
            },
            "rooms": rooms_json
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _add_boundary_detection(self, enhanced):
        """改进版墙体/边界提取 (V2+ 扩展)"""
        print("🔲 边界重构(V2+ 扩展: 轮廓简化 + 直线拟合)...")
        arr = enhanced.copy()
        room_labels = {2,3,4,6,7,8}
        wall_labels = {9,10}
        original_wall_mask = np.isin(arr, list(wall_labels))

        # 1) 平滑 & 极小碎片过滤 + 轮廓简化
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        smoothed = arr.copy()
        MIN_ROOM_COMPONENT = 25
        for lbl in room_labels:
            mask = (arr==lbl).astype(np.uint8)
            if mask.sum()==0:
                continue
            num_c, lab_c, stats_c, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            for cid in range(1, num_c):
                if stats_c[cid, cv2.CC_STAT_AREA] < MIN_ROOM_COMPONENT:
                    mask[lab_c==cid] = 0
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
            try:
                contours,_ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                simp = np.zeros_like(opened)
                for cnt in contours:
                    peri = cv2.arcLength(cnt, True)
                    eps = max(1.0, 0.005*peri)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    cv2.fillPoly(simp,[approx],1)
                opened = simp
            except Exception:
                pass
            smoothed[opened==1] = lbl
        smoothed[original_wall_mask] = arr[original_wall_mask]

        # 2) 标签差分
        padded = np.pad(smoothed,1,mode='edge')
        center = padded[1:-1,1:-1]
        up = padded[0:-2,1:-1]; down = padded[2:,1:-1]; left = padded[1:-1,0:-2]; right = padded[1:-1,2:]
        def diff_mask(neigh):
            return (center!=neigh) & (np.isin(center,list(room_labels)) | np.isin(neigh,list(room_labels)) | (center==0) | (neigh==0))
        boundary_candidates = diff_mask(up)|diff_mask(down)|diff_mask(left)|diff_mask(right)
        candidate_mask = boundary_candidates.astype(np.uint8)

        # 3) 细化
        def zhang_suen_thinning(bin_img):
            img = bin_img.copy().astype(np.uint8)
            changed=True
            while changed:
                changed=False
                to_remove=[]
                for y in range(1,img.shape[0]-1):
                    for x in range(1,img.shape[1]-1):
                        if img[y,x]==1:
                            P2=img[y-1,x];P3=img[y-1,x+1];P4=img[y,x+1];P5=img[y+1,x+1];P6=img[y+1,x];P7=img[y+1,x-1];P8=img[y,x-1];P9=img[y-1,x-1]
                            nbr=[P2,P3,P4,P5,P6,P7,P8,P9]; cnt=sum(nbr)
                            if 2<=cnt<=6:
                                trans=0
                                for i in range(8):
                                    if nbr[i]==0 and nbr[(i+1)%8]==1: trans+=1
                                if trans==1 and (P2*P4*P6)==0 and (P4*P6*P8)==0: to_remove.append((y,x))
                if to_remove:
                    changed=True
                    for y,x in to_remove: img[y,x]=0
                to_remove=[]
                for y in range(1,img.shape[0]-1):
                    for x in range(1,img.shape[1]-1):
                        if img[y,x]==1:
                            P2=img[y-1,x];P3=img[y-1,x+1];P4=img[y,x+1];P5=img[y+1,x+1];P6=img[y+1,x];P7=img[y+1,x-1];P8=img[y,x-1];P9=img[y-1,x-1]
                            nbr=[P2,P3,P4,P5,P6,P7,P8,P9]; cnt=sum(nbr)
                            if 2<=cnt<=6:
                                trans=0
                                for i in range(8):
                                    if nbr[i]==0 and nbr[(i+1)%8]==1: trans+=1
                                if trans==1 and (P2*P4*P8)==0 and (P2*P6*P8)==0: to_remove.append((y,x))
                if to_remove:
                    changed=True
                    for y,x in to_remove: img[y,x]=0
            return img
        skeleton = zhang_suen_thinning(candidate_mask)

        # Hough 补强
        try:
            ls = (candidate_mask*255).astype(np.uint8)
            lines = cv2.HoughLinesP(ls,1,np.pi/180,threshold=18,minLineLength=14,maxLineGap=4)
            if lines is not None:
                hmask = np.zeros_like(skeleton)
                for l in lines[:1500]:
                    x1,y1,x2,y2 = l[0]; cv2.line(hmask,(x1,y1),(x2,y2),1,1)
                skeleton = ((skeleton==1)|(hmask==1)).astype(np.uint8)
                print(f"   📐 Hough 直线补强: {len(lines)} 条")
        except Exception as _e:
            print(f"   ⚠️ Hough 跳过: {_e}")

        # 4) 吸附与加粗
        ORTHO_TOL=15; allow_diagonal=True
        skel_arr=skeleton.copy(); new_skel=np.zeros_like(skel_arr)
        num_s,lab_s,stats_s,_ = cv2.connectedComponentsWithStats(skel_arr.astype(np.uint8),connectivity=8)
        for sid in range(1,num_s):
            comp=(lab_s==sid); ys,xs=np.where(comp)
            if xs.size<3: new_skel[comp]=1; continue
            xc=xs.mean(); yc=ys.mean(); xz=xs-xc; yz=ys-yc
            cov=np.cov(np.vstack((xz,yz))); eigvals,eigvecs=np.linalg.eig(cov)
            vx,vy=eigvecs[:,np.argmax(eigvals)]; angle=(np.degrees(np.arctan2(vy,vx))+180)%180
            if angle<ORTHO_TOL or angle>180-ORTHO_TOL:
                yline=int(round(yc)); new_skel[yline, xs.min():xs.max()+1]=1
            elif abs(angle-90)<ORTHO_TOL:
                xline=int(round(xc)); new_skel[ys.min():ys.max()+1, xline]=1
            else:
                if allow_diagonal: new_skel[comp]=1
                else:
                    for ux in np.unique(xs):
                        yv=ys[xs==ux]; new_skel[int(np.median(yv)), ux]=1
        skeleton=new_skel

        if np.any(skeleton):
            dil=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            thick=cv2.dilate(skeleton.astype(np.uint8),dil,iterations=1)
        else:
            thick=skeleton

        def merge_small_gaps(m):
            merged=m.copy()
            # 行
            for y in range(merged.shape[0]):
                xs=np.where(merged[y]==1)[0]
                if xs.size==0: continue
                pv=xs[0]
                for x in xs[1:]:
                    gap=x-pv-1
                    if 0<gap<=2: merged[y,pv+1:x]=1
                    pv=x
            # 列
            for x in range(merged.shape[1]):
                ys=np.where(merged[:,x]==1)[0]
                if ys.size==0: continue
                pv=ys[0]
                for y in ys[1:]:
                    gap=y-pv-1
                    if 0<gap<=2: merged[pv+1:y,x]=1
                    pv=y
            return merged
        thick=merge_small_gaps(thick.astype(np.uint8))

        def close_endpoints(m):
            lm=m.copy(); H,W=lm.shape; dirs=[(1,0),(-1,0),(0,1),(0,-1)]
            endpoints=[]
            for y in range(H):
                for x in range(W):
                    if lm[y,x]==1:
                        deg=0
                        for dx,dy in dirs:
                            nx,ny=x+dx,y+dy
                            if 0<=nx<W and 0<=ny<H and lm[ny,nx]==1: deg+=1
                        if deg==1: endpoints.append((x,y))
            used=set()
            for i,(x1,y1) in enumerate(endpoints):
                if i in used: continue
                best=None; bestd=1e9
                for j,(x2,y2) in enumerate(endpoints):
                    if j<=i or j in used: continue
                    dx=abs(x2-x1); dy=abs(y2-y1)
                    if max(dx,dy)<=5 and ((y1==y2) or (x1==x2) or (dx+dy)<=5):
                        d=dx+dy
                        if d<bestd: bestd=d; best=j
                if best is not None:
                    x2,y2=endpoints[best]
                    if y1==y2: lm[y1,min(x1,x2):max(x1,x2)+1]=1
                    elif x1==x2: lm[min(y1,y2):max(y1,y2)+1,x1]=1
                    else:
                        lm[y1,min(x1,x2):max(x1,x2)+1]=1
                        lm[min(y1,y2):max(y1,y2)+1,x2]=1
                    used.add(i); used.add(best)
            return lm
        thick=close_endpoints(thick)

        # 5) 墙体瘦身 + 骨架融合
        wall_mask_initial=(arr==10).astype(np.uint8)
        num,labels_im,stats,_=cv2.connectedComponentsWithStats(wall_mask_initial,connectivity=4)
        large_perimeter_mask=np.zeros_like(wall_mask_initial)
        for comp in range(1,num):
            area=stats[comp,cv2.CC_STAT_AREA]
            if area>400:
                comp_mask=(labels_im==comp).astype(np.uint8)
                eroded=cv2.erode(comp_mask,kernel,iterations=1)
                perimeter=comp_mask-eroded
                large_perimeter_mask[perimeter==1]=1
            else:
                large_perimeter_mask[labels_im==comp]=1

        new_arr=arr.copy()
        new_arr[new_arr==10]=0
        new_arr[large_perimeter_mask==1]=10
        add_mask=(thick==1) & (~np.isin(new_arr,[9,10]))
        new_arr[add_mask]=10

        # 6) 噪点清理
        wall_mask_final=(new_arr==10).astype(np.uint8)
        num2,labels2,stats2,_=cv2.connectedComponentsWithStats(wall_mask_final,connectivity=8)
        removed=0
        for comp in range(1,num2):
            area=stats2[comp,cv2.CC_STAT_AREA]
            if area<3:
                removed+=area
                new_arr[labels2==comp]=0
        added=int(add_mask.sum())
        # 7) 竖直整列伪墙抑制 (几乎全高且孤立的细列)
        H,W=new_arr.shape; removed_cols=0
        col_wall_ratio = []
        for cx in range(W):
            col_vals = new_arr[:,cx]
            wall_ratio = (col_vals==10).mean()
            col_wall_ratio.append(wall_ratio)
        import numpy as _np
        col_wall_ratio = _np.array(col_wall_ratio)
        # 计算左右相邻平均，判断孤立
        for cx in range(W):
            wr = col_wall_ratio[cx]
            if wr>0.95:  # 几乎整列墙
                left_wr = col_wall_ratio[cx-1] if cx-1>=0 else 1.0
                right_wr = col_wall_ratio[cx+1] if cx+1<W else 1.0
                # 两侧都不是大比例墙体，说明突兀
                if left_wr<0.30 and right_wr<0.30:
                    new_arr[new_arr[:,cx]==10, cx]=0
                    removed_cols+=1
        if removed_cols>0:
            print(f"🛠️ 伪竖墙列抑制: 移除 {removed_cols} 列接近全高的孤立竖线")
        print(f"✅ 边界重构完成: 新增墙体 {added} 像素, 清理噪点 {removed} 像素, 伪列移除 {removed_cols} 列, 大块组件 {num-1} -> {np.unique(labels2).size-1}")
        return new_arr

    def _add_boundary_detection_cached(self, enhanced):
        """带缓存包装，避免同一 label 图多次细化导致碎裂"""
        try:
            import hashlib
            key = hashlib.md5(enhanced.tobytes()).hexdigest()
        except Exception:
            key = str(id(enhanced))
        if key in self._boundary_cache:
            return self._boundary_cache[key]
        result = self._add_boundary_detection(enhanced)
        self._boundary_cache[key] = result
        return result

    # ===== 矢量化: 从墙体二值图提取线段并导出 SVG =====
    def _vectorize_walls(self, boundary_mask_binary, original_size):
        """返回字典: {walls_segments, walls_polylines, openings_segments} (均已缩放到原尺寸)"""
        import numpy as np, cv2, math
        ow,oh = original_size
        wall_mask_512 = (boundary_mask_binary==255).astype(np.uint8)
        open_mask_512 = (boundary_mask_binary==128).astype(np.uint8)

        def skeletonize(mask):
            skel = mask.copy()
            def thin_once(img):
                h,w=img.shape; remove=[]
                for y in range(1,h-1):
                    for x in range(1,w-1):
                        if img[y,x]==1:
                            P2=img[y-1,x];P3=img[y-1,x+1];P4=img[y,x+1];P5=img[y+1,x+1];P6=img[y+1,x];P7=img[y+1,x-1];P8=img[y,x-1];P9=img[y-1,x-1]
                            nbr=[P2,P3,P4,P5,P6,P7,P8,P9]; cnt=sum(nbr)
                            if 2<=cnt<=6:
                                trans=0
                                for i in range(8):
                                    if nbr[i]==0 and nbr[(i+1)%8]==1: trans+=1
                                if trans==1 and (P2*P4*P6)==0 and (P4*P6*P8)==0:
                                    remove.append((y,x))
                for (y,x) in remove: img[y,x]=0
                return len(remove)>0
            it=0
            while thin_once(skel) and it<8: it+=1
            return skel

        def chains_from_skel(skel):
            h,w=skel.shape; visited=np.zeros_like(skel,bool)
            dirs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            chains=[]
            for y in range(h):
                for x in range(w):
                    if skel[y,x]==1 and not visited[y,x]:
                        stack=[(x,y)]; visited[y,x]=True; pts=[]
                        while stack:
                            cx,cy=stack.pop(); pts.append((cx,cy))
                            for dx,dy in dirs:
                                nx,ny=cx+dx,cy+dy
                                if 0<=nx<w and 0<=ny<h and skel[ny,nx]==1 and not visited[ny,nx]:
                                    visited[ny,nx]=True; stack.append((nx,ny))
                        if len(pts)>=3: chains.append(pts)
            return chains

        def rdp(points, eps=1.8):
            if len(points)<3: return points
            x1,y1=points[0]; x2,y2=points[-1]; A=np.array([x2-x1,y2-y1]); L=np.linalg.norm(A)
            if L==0: dists=[0]*len(points)
            else:
                dists=[]
                for (x,y) in points:
                    v=np.array([x-x1,y-y1]); proj=(v@A)/L if L else 0
                    proj_pt=np.array([x1,y1]) + (proj/L)*A if L else np.array([x1,y1])
                    dists.append(np.linalg.norm(np.array([x,y])-proj_pt))
            idx=int(np.argmax(dists))
            if dists[idx]>eps:
                return rdp(points[:idx+1],eps)[:-1]+rdp(points[idx:],eps)
            return [points[0],points[-1]]

        def chains_to_segments(chains):
            segs=[]
            for c in chains:
                c_sorted=sorted(c)
                simp=rdp(c_sorted)
                for i in range(len(simp)-1):
                    x1,y1=simp[i]; x2,y2=simp[i+1]
                    if (x1,y1)!=(x2,y2): segs.append((x1,y1,x2,y2))
            return segs

        def merge_colinear(segments, ang_tol=6, dist_tol=4):
            segments=list(segments); merged=True
            def colinear(a,b):
                x1,y1,x2,y2=a; x3,y3,x4,y4=b
                v1=np.array([x2-x1,y2-y1]); v2=np.array([x4-x3,y4-y3])
                n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
                if n1==0 or n2==0: return False
                ang=math.degrees(math.acos(np.clip((v1@v2)/(n1*n2),-1,1)))
                if ang>ang_tol: return False
                for p in [(x1,y1),(x2,y2)]:
                    for q in [(x3,y3),(x4,y4)]:
                        if ( (p[0]-q[0])**2+(p[1]-q[1])**2 )**0.5 < dist_tol: return True
                return False
            while merged:
                merged=False; out=[]; used=[False]*len(segments)
                for i,a in enumerate(segments):
                    if used[i]: continue
                    ax1,ay1,ax2,ay2=a; vax=np.array([ax2-ax1, ay2-ay1])
                    for j,b in enumerate(segments):
                        if j<=i or used[j]: continue
                        if colinear(a,b):
                            bx1,by1,bx2,by2=b
                            pts=[(ax1,ay1),(ax2,ay2),(bx1,by1),(bx2,by2)]
                            if abs(vax[0])>=abs(vax[1]): pts=sorted(pts,key=lambda p:p[0])
                            else: pts=sorted(pts,key=lambda p:p[1])
                            ax1,ay1=pts[0]; ax2,ay2=pts[-1]
                            used[j]=True; merged=True
                    used[i]=True; out.append((ax1,ay1,ax2,ay2))
                segments=out
            return segments

        def angle_snap(segments, ang_set=(0,45,90,135), tol=12):
            snapped=[]
            for (x1,y1,x2,y2) in segments:
                dx=x2-x1; dy=y2-y1
                if dx==0 and dy==0: continue
                ang= (math.degrees(math.atan2(dy,dx)) + 180) % 180
                best=None; best_diff=999
                for a in ang_set:
                    diff=min(abs(ang-a), 180-abs(ang-a))
                    if diff<best_diff: best_diff=diff; best=a
                if best_diff<=tol:
                    # 以中点 & 原长度重新构造
                    L=(dx*dx+dy*dy)**0.5
                    cx=(x1+x2)/2; cy=(y1+y2)/2
                    rad=math.radians(best)
                    hx=(L/2)*math.cos(rad); hy=(L/2)*math.sin(rad)
                    nx1=cx-hx; ny1=cy-hy; nx2=cx+hx; ny2=cy+hy
                    snapped.append((int(round(nx1)),int(round(ny1)),int(round(nx2)),int(round(ny2))))
                else:
                    snapped.append((x1,y1,x2,y2))
            return snapped

        def segments_to_polylines(segments, join_tol=4):
            # 构造端点图
            pts=[]
            for (x1,y1,x2,y2) in segments: pts.extend([(x1,y1),(x2,y2)])
            # 去重
            uniq=[]
            for p in pts:
                if not any((abs(p[0]-q[0])<=1 and abs(p[1]-q[1])<=1) for q in uniq): uniq.append(p)
            # 映射
            def find_idx(p):
                for i,q in enumerate(uniq):
                    if abs(p[0]-q[0])<=1 and abs(p[1]-q[1])<=1: return i
                uniq.append(p); return len(uniq)-1
            adj={i:set() for i in range(len(uniq))}
            for (x1,y1,x2,y2) in segments:
                i1=find_idx((x1,y1)); i2=find_idx((x2,y2))
                adj[i1].add(i2); adj[i2].add(i1)
            polylines=[]; visited=set()
            for start in adj:
                if start in visited: continue
                # 线性链/环遍历
                stack=[start]; current=[]
                while stack:
                    v=stack.pop();
                    if v in visited: continue
                    visited.add(v); current.append(uniq[v])
                    for nb in adj[v]:
                        if nb not in visited: stack.append(nb)
                if len(current)>=2: polylines.append(sorted(current))
            return polylines

        def scale_segments(segs):
            scaled=[]
            for (x1,y1,x2,y2) in segs:
                sx1=int(round(x1/512*ow)); sy1=int(round(y1/512*oh))
                sx2=int(round(x2/512*ow)); sy2=int(round(y2/512*oh))
                if (sx1,sy1)!=(sx2,sy2): scaled.append((sx1,sy1,sx2,sy2))
            return scaled

        # ========== 墙体处理 ==========
        wall_skel = skeletonize(wall_mask_512)
        wall_chains = chains_from_skel(wall_skel)
        wall_segments = chains_to_segments(wall_chains)
        wall_segments = merge_colinear(angle_snap(wall_segments))
        wall_polylines = segments_to_polylines(wall_segments)
        wall_segments_scaled = scale_segments(wall_segments)

        # ========== 开口处理（简单：直接骨架提取） ==========
        open_skel = skeletonize(open_mask_512) if open_mask_512.sum()>0 else open_mask_512
        open_chains = chains_from_skel(open_skel) if open_mask_512.sum()>0 else []
        open_segments = chains_to_segments(open_chains)
        open_segments = merge_colinear(angle_snap(open_segments))
        open_segments_scaled = scale_segments(open_segments)

        return {
            'walls_segments': wall_segments_scaled,
            'walls_polylines': wall_polylines,  # 未缩放点集合(512坐标)可后续利用
            'openings_segments': open_segments_scaled
        }

    def _export_walls_svg(self, path, vec_data, original_size):
        ow,oh = original_size
        walls = vec_data['walls_segments']
        openings = vec_data['openings_segments']
        lines=[
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{ow}" height="{oh}" viewBox="0 0 {ow} {oh}" stroke-linecap="round" stroke-linejoin="round">',
            '<g id="walls" stroke="#000" stroke-width="2" fill="none">'
        ]
        for (x1,y1,x2,y2) in walls:
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        lines.append('</g>')
        lines.append('<g id="openings" stroke="#FF3C80" stroke-width="2" fill="none">')
        for (x1,y1,x2,y2) in openings:
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        lines.append('</g></svg>')
        with open(path,'w',encoding='utf-8') as f: f.write('\n'.join(lines))

    def _export_walls_dxf(self, path, vec_data, original_size):
        ow,oh = original_size
        walls = vec_data['walls_segments']
        openings = vec_data['openings_segments']
        def dxf_header():
            return ["0","SECTION","2","HEADER","0","ENDSEC","0","SECTION","2","ENTITIES"]
        def dxf_footer():
            return ["0","ENDSEC","0","EOF"]
        lines=dxf_header()
        # 墙体 LINE
        for (x1,y1,x2,y2) in walls:
            lines += ["0","LINE","8","WALLS","10",str(x1),"20",str(y1),"11",str(x2),"21",str(y2)]
        # 开口 LINE (图层 OPEN)
        for (x1,y1,x2,y2) in openings:
            lines += ["0","LINE","8","OPEN","10",str(x1),"20",str(y1),"11",str(x2),"21",str(y2)]
        lines += dxf_footer()
        with open(path,'w',encoding='utf-8') as f: f.write('\n'.join(lines))

    def _extract_room_coordinates(
        self, enhanced_resized, original_size, room_text_items
    ):
        """提取各房间的坐标信息，优先使用OCR文字位置，支持多个同类型房间"""
        # ===== A + C 预处理: 针对厨房的小碎片合并 (C) =====
        # 场景: 分割输出厨房 label(7) 可能被墙线割裂成多个很小碎片, 导致后续基于单个 OCR seed 的 BFS 只抓到一小块。
        # 策略(C): 若厨房总面积占比很小(<1.2%) 且连通域数量>1, 对厨房掩膜做一次温和闭运算+膨胀以桥接近距离碎片。
        kitchen_fragment_merged_mask = None
        try:
            mask_k = (enhanced_resized == 7).astype(np.uint8)
            total_pixels_512 = enhanced_resized.shape[0] * enhanced_resized.shape[1]
            kitchen_pixels = int(mask_k.sum())
            if kitchen_pixels > 0:
                area_ratio_k = kitchen_pixels / float(total_pixels_512)
                num_k, lab_k, stats_k, _ = cv2.connectedComponentsWithStats(mask_k, connectivity=4)
                comp_cnt = num_k - 1
                if area_ratio_k < 0.012 and comp_cnt > 1:
                    # 统计小碎片数量
                    small_components = sum(1 for cid in range(1, num_k) if stats_k[cid, cv2.CC_STAT_AREA] < 160)
                    if small_components >= 1:
                        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        merged1 = cv2.morphologyEx(mask_k, cv2.MORPH_CLOSE, k_close, iterations=1)
                        merged2 = cv2.dilate(merged1, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                        # 仅在膨胀后新增像素不超过原厨房面积的 60% 时接受（防止误吞其它区域）
                        added = merged2.sum() - mask_k.sum()
                        if added <= kitchen_pixels * 0.6:
                            kitchen_fragment_merged_mask = merged2
                            print(f"🧩 [碎片合并C] 厨房碎片数={comp_cnt} 小碎片={small_components} 面积占比={area_ratio_k:.2%} -> 应用闭运算+膨胀 合并增量像素={int(added)}")
                        else:
                            print(f"🧩 [碎片合并C] 拒绝合并: 新增像素过多({int(added)} > {int(kitchen_pixels*0.6)})")
        except Exception as _frag_e:
            print(f"⚠️ [碎片合并C] 异常: {_frag_e}")
        room_info = {}

        # 计算坐标转换比例
        original_width, original_height = original_size

        # 定义房间类型及其在分割掩码中的标签
        room_types = ["厨房", "卫生间", "客厅", "卧室", "阳台", "书房"]
        room_label_mapping = {
            "厨房": 7,
            "卫生间": 2,
            "客厅": 3,
            "卧室": 4,
            "阳台": 6,
            "书房": 8,
        }

        # 初始化所有房间信息为空列表，支持多个同类型房间
        for room_type in room_types:
            room_info[room_type] = []

        # 优先使用OCR文字位置确定房间坐标
        for item in room_text_items:
            text = item["text"].lower().strip()

            # 匹配房间类型
            room_type = None
            if any(keyword in text for keyword in ["厨房", "kitchen", "厨"]):
                room_type = "厨房"
            elif any(
                keyword in text
                for keyword in [
                    "卫生间",
                    "bathroom",
                    "卫",
                    "洗手间",
                    "浴室",
                    "淋浴间",
                    "shower",
                    "淋浴",
                    "盥洗室",
                ]
            ):
                room_type = "卫生间"
            elif any(keyword in text for keyword in ["客厅", "living", "厅", "起居室"]):
                room_type = "客厅"
            elif any(
                keyword in text for keyword in ["卧室", "bedroom", "主卧", "次卧", "卧房", "卧空", "网房"]
            ):
                # 扩展卧室同义词/误识别词支持（网房/卧空等）
                room_type = "卧室"
            elif any(keyword in text for keyword in ["阳台", "balcony", "阳兮", "阳合", "阳囊"]):
                room_type = "阳台"
                if text in ["阳兮", "阳合", "阳囊"]:
                    print(f"🔧 [OCR修正] 误识别'{text}' -> '阳台'")
            elif any(
                keyword in text
                for keyword in ["书房", "study", "办公室", "office"]
            ):
                room_type = "书房"
                print(f"🔍 [OCR验证] 确认书房: '{text}' (OCR支持)")

            if room_type and room_type in room_info:
                # 使用OCR文字的 bbox (已在 ocr_enhanced 中缩放回原图坐标)
                x, y, w, h = item["bbox"]

                # 计算文字中心 (原图坐标)
                ocr_center_x = x + w // 2
                ocr_center_y = y + h // 2

                # 早期版本存在再次除以 scale_factor 的错误 (导致坐标偏小 0.5x)
                # 修正: 直接使用当前中心 (假定 bbox 已是原尺度)
                orig_center_x = ocr_center_x
                orig_center_y = ocr_center_y

                # 保护性检测: 如果 bbox 明显超出原图尺寸 (>1.2x), 说明可能还没缩放, 再按 scale_factor 回调
                scale_factor = float(item.get('scale_factor', 1.0) or 1.0)
                if scale_factor > 1.01 and (orig_center_x > original_width * 1.2 or orig_center_y > original_height * 1.2):
                    orig_center_x = int(round(orig_center_x / scale_factor))
                    orig_center_y = int(round(orig_center_y / scale_factor))
                    x = int(round(x / scale_factor))
                    y = int(round(y / scale_factor))
                    w = int(round(w / scale_factor))
                    h = int(round(h / scale_factor))
                    print(f"🔧 [坐标自适应] 发现未缩放OCR框, 已按 scale_factor={scale_factor:.2f} 回调 -> center=({orig_center_x},{orig_center_y})")

                # 优先使用分割掩码确定整间房的边界
                min_x = max_x = min_y = max_y = None
                label = room_label_mapping.get(room_type)
                if label is not None:
                    # 使用碎片合并后的厨房掩膜 (C)
                    if label == 7 and kitchen_fragment_merged_mask is not None:
                        mask = kitchen_fragment_merged_mask.astype(bool)
                    else:
                        mask = (enhanced_resized == label)
                    mask_h, mask_w = mask.shape
                    # 将原图中心映射到 512 掩膜坐标
                    mask_x = int(round(orig_center_x * mask_w / original_width))
                    mask_y = int(round(orig_center_y * mask_h / original_height))
                    seed_x, seed_y, seed_found = mask_x, mask_y, False

                    # 边界保护
                    if not (0 <= mask_x < mask_w and 0 <= mask_y < mask_h):
                        mask_x = min(max(mask_x, 0), mask_w - 1)
                        mask_y = min(max(mask_y, 0), mask_h - 1)
                        seed_x, seed_y = mask_x, mask_y

                    # 如果中心点不是该标签，扩大搜索半径寻找最近的同标签像素
                    if mask[mask_y, mask_x]:
                        seed_found = True
                    else:
                        for search_radius in (3, 6, 10):  # 分阶段扩大
                            min_dist = None
                            for dy in range(-search_radius, search_radius + 1):
                                for dx in range(-search_radius, search_radius + 1):
                                    nx, ny = mask_x + dx, mask_y + dy
                                    if 0 <= nx < mask_w and 0 <= ny < mask_h and mask[ny, nx]:
                                        dist = dx * dx + dy * dy
                                        if min_dist is None or dist < min_dist:
                                            min_dist = dist
                                            seed_x, seed_y = nx, ny
                                            seed_found = True
                            if seed_found:
                                break

                    if seed_found:
                        labeled_mask = mask.astype(np.uint8)
                        num_labels, labels_img = cv2.connectedComponents(labeled_mask)
                        region_label = labels_img[seed_y, seed_x]
                        if region_label != 0:
                            full_region = (labels_img == region_label)

                            # ===== 半径限制泛洪，避免一个标签吞并多个逻辑房间 =====
                            # 根据房间类型设定最大半径 (原图像素) 与最大面积比上限
                            max_radius_map = {"厨房": 180, "卫生间": 160, "卧室": 260, "阳台": 220, "书房": 240, "客厅": 480}
                            max_area_ratio_map = {"厨房": 0.15, "卫生间": 0.12, "卧室": 0.28, "阳台": 0.20, "书房": 0.22, "客厅": 0.38}
                            max_radius_orig = max_radius_map.get(room_type, 300)
                            max_area_ratio = max_area_ratio_map.get(room_type, 0.30)

                            # 转换到 512 空间的最大半径
                            radius_512_x = int(round(max_radius_orig * mask_w / original_width))
                            radius_512_y = int(round(max_radius_orig * mask_h / original_height))
                            radius_512 = int((radius_512_x + radius_512_y) / 2)

                            # BFS 受限泛洪
                            visited = np.zeros_like(full_region, dtype=np.uint8)
                            from collections import deque
                            q = deque()
                            q.append((seed_x, seed_y))
                            visited[seed_y, seed_x] = 1
                            sel_pixels = [(seed_x, seed_y)]
                            while q:
                                cx, cy = q.popleft()
                                for nx in (cx-1, cx, cx+1):
                                    for ny in (cy-1, cy, cy+1):
                                        if nx == cx and ny == cy: continue
                                        if 0 <= nx < mask_w and 0 <= ny < mask_h and not visited[ny, nx]:
                                            if full_region[ny, nx]:
                                                # 半径约束
                                                if abs(nx - seed_x) <= radius_512 and abs(ny - seed_y) <= radius_512:
                                                    visited[ny, nx] = 1
                                                    q.append((nx, ny))
                                                    sel_pixels.append((nx, ny))
                                            visited[ny, nx] = 1  # 标记访问避免重复

                            sel_pixels_arr = np.array(sel_pixels)
                            x_coords = sel_pixels_arr[:,0]
                            y_coords = sel_pixels_arr[:,1]
                            min_x_512, max_x_512 = x_coords.min(), x_coords.max()
                            min_y_512, max_y_512 = y_coords.min(), y_coords.max()

                            # 如果选择区域面积比超过最大限制或区域太小与完整区域面积差异巨大，退回使用完整区域
                            limited_area = len(sel_pixels)
                            full_area = int(full_region.sum())
                            total_pixels = original_width * original_height
                            bbox_area_est = (max_x_512 - min_x_512 + 1) * (max_y_512 - min_y_512 + 1) * (total_pixels / (mask_w * mask_h))
                            if (bbox_area_est / total_pixels) > max_area_ratio or limited_area < min(50, full_area * 0.05):
                                # 使用原完整区域
                                y_coords, x_coords = np.where(full_region)
                                min_x_512, max_x_512 = x_coords.min(), x_coords.max()
                                min_y_512, max_y_512 = y_coords.min(), y_coords.max()
                                print(f"⚠️ [房间裁剪] {room_type} 受限泛洪不稳定(面积或尺寸异常)，回退使用完整连通域")
                            else:
                                print(f"✅ [房间裁剪] {room_type} 受限泛洪选取 {limited_area}/{full_area} 像素, 避免过度扩张")

                            scale_x = original_width / float(mask_w)
                            scale_y = original_height / float(mask_h)
                            min_x = int(min_x_512 * scale_x)
                            max_x = int(max_x_512 * scale_x)
                            min_y = int(min_y_512 * scale_y)
                            max_y = int(max_y_512 * scale_y)

                            # 过大区域保护: 若 bbox 占原图面积 > 40% (无 OCR 情况除外)，认为泛化过度，尝试局部收缩
                            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                            whole_area = original_width * original_height
                            if bbox_area / whole_area > 0.40 and text not in ("分割检测",):
                                # 在 512 空间构造距离中心点的局部窗口 (限制 35% 原始宽/高)
                                win_w = int(min(mask_w * 0.5, max(64, mask_w * 0.35)))
                                win_h = int(min(mask_h * 0.5, max(64, mask_h * 0.35)))
                                cx512 = int(round(orig_center_x * mask_w / original_width))
                                cy512 = int(round(orig_center_y * mask_h / original_height))
                                x1_l = max(0, cx512 - win_w // 2)
                                x2_l = min(mask_w, cx512 + win_w // 2)
                                y1_l = max(0, cy512 - win_h // 2)
                                y2_l = min(mask_h, cy512 + win_h // 2)
                                local = full_region[y1_l:y2_l, x1_l:x2_l]
                                if local.any():
                                    ly, lx = np.where(local)
                                    # 映射回全局 512
                                    min_x_512 = x1_l + lx.min(); max_x_512 = x1_l + lx.max()
                                    min_y_512 = y1_l + ly.min(); max_y_512 = y1_l + ly.max()
                                    min_x = int(min_x_512 * scale_x)
                                    max_x = int(max_x_512 * scale_x)
                                    min_y = int(min_y_512 * scale_y)
                                    max_y = int(max_y_512 * scale_y)
                                    print(f"⚠️ [坐标调整] {room_type} 区域过大({bbox_area/whole_area:.1%}), 使用局部窗口收缩 bbox")
                            print(f"🧩 [坐标调试] {room_type} OCR中心=({orig_center_x},{orig_center_y}) 种子=({seed_x},{seed_y}) bbox=({min_x},{min_y},{max_x},{max_y})")

                if min_x is None:
                    # 未找到连通域，回退到基于OCR文字自身的最小边界 (bbox 已是原图尺度)
                    orig_w = w
                    orig_h = h
                    # 给一些冗余避免过紧裁剪
                    half_width = max(20, int(orig_w * 0.6))
                    half_height = max(15, int(orig_h * 0.6))
                    min_x = max(0, orig_center_x - half_width)
                    max_x = min(original_width - 1, orig_center_x + half_width)
                    min_y = max(0, orig_center_y - half_height)
                    max_y = min(original_height - 1, orig_center_y + half_height)

                width = max_x - min_x + 1
                height = max_y - min_y + 1

                room_info[room_type].append({
                    'center': (orig_center_x, orig_center_y),
                    'bbox': (min_x, min_y, max_x, max_y),
                    'pixels': width * height,  # 基于边界框的面积 (后续可替换为真实 mask 面积)
                    'width': width,
                    'height': height,
                    'text': text,
                    'raw_text': item.get('raw_text', item.get('text', '')),
                    'confidence': item.get('confidence', 0.0),
                    'source': 'ocr'
                })

        # ===== A: 基于 OCR seed 的厨房区域重建 / 扩展 =====
        try:
            if room_info.get('厨房'):
                orig_w, orig_h = original_size
                img_area = orig_w * orig_h
                rebuilt_any = False
                new_kitchens = []
                for k_room in room_info['厨房']:
                    bx1, by1, bx2, by2 = k_room['bbox']
                    bbox_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
                    area_ratio = bbox_area / img_area
                    need_rebuild = (area_ratio < 0.006) or (k_room['width'] < 55) or (k_room['height'] < 40)
                    if not need_rebuild:
                        new_kitchens.append(k_room)
                        continue
                    # 进入重建: 在 512 空间收集附近 label=7 像素 (使用合并掩膜若可用)
                    mask512 = kitchen_fragment_merged_mask if kitchen_fragment_merged_mask is not None else (enhanced_resized == 7).astype(np.uint8)
                    if mask512.sum() == 0:
                        # 无分割支持 -> 直接 OCR 中心扩展到目标面积
                        cx, cy = k_room['center']
                        target_ratio = 0.022  # 2.2%
                        target_area = int(img_area * target_ratio)
                        side = int((target_area) ** 0.5)
                        half = side // 2
                        nx1 = max(0, cx - half)
                        nx2 = min(orig_w - 1, cx + half)
                        ny1 = max(0, cy - half)
                        ny2 = min(orig_h - 1, cy + half)
                        k_room.update({'bbox': (nx1, ny1, nx2, ny2), 'width': nx2-nx1+1, 'height': ny2-ny1+1, 'pixels': (nx2-nx1+1)*(ny2-ny1+1), 'rebuild': 'ocr_expand_no_seg'})
                        rebuilt_any = True
                        print(f"🍳 [厨房重建A] 无分割厨房: OCR 扩展为 {(nx2-nx1+1)}x{(ny2-ny1+1)} 面积比={(nx2-nx1+1)*(ny2-ny1+1)/img_area:.2%}")
                        new_kitchens.append(k_room)
                        continue
                    # 有分割: 映射 OCR 中心到 512
                    cx, cy = k_room['center']
                    cx512 = int(round(cx / orig_w * mask512.shape[1]))
                    cy512 = int(round(cy / orig_h * mask512.shape[0]))
                    # 多级窗口扩展收集 label=7 像素
                    collected = None
                    for win in (40, 60, 80, 100):
                        x1 = max(0, cx512 - win)
                        x2 = min(mask512.shape[1]-1, cx512 + win)
                        y1 = max(0, cy512 - win)
                        y2 = min(mask512.shape[0]-1, cy512 + win)
                        sub = mask512[y1:y2+1, x1:x2+1]
                        if sub.sum() == 0:
                            continue
                        collected = (x1, y1, x2, y2, sub.copy())
                        # 如果子窗口内厨房像素占窗口 > 9% 或像素数量 > 350 即可停止扩大
                        if (sub.sum() / ((x2-x1+1)*(y2-y1+1))) > 0.09 or sub.sum() > 350:
                            break
                    if collected is None:
                        # 退回 OCR 扩展
                        target_ratio = 0.022
                        target_area = int(img_area * target_ratio)
                        side = int((target_area) ** 0.5)
                        half = side // 2
                        nx1 = max(0, cx - half)
                        nx2 = min(orig_w - 1, cx + half)
                        ny1 = max(0, cy - half)
                        ny2 = min(orig_h - 1, cy + half)
                        k_room.update({'bbox': (nx1, ny1, nx2, ny2), 'width': nx2-nx1+1, 'height': ny2-ny1+1, 'pixels': (nx2-nx1+1)*(ny2-ny1+1), 'rebuild': 'ocr_expand_no_pixels'})
                        rebuilt_any = True
                        print(f"🍳 [厨房重建A] 分割窗口无像素: OCR 扩展为 {(nx2-nx1+1)}x{(ny2-ny1+1)} 面积比={(nx2-nx1+1)*(ny2-ny1+1)/img_area:.2%}")
                        new_kitchens.append(k_room)
                        continue
                    x1_512, y1_512, x2_512, y2_512, sub = collected
                    ys, xs = np.where(sub > 0)
                    if len(xs) == 0:
                        new_kitchens.append(k_room)
                        continue
                    minx = x1_512 + xs.min(); maxx = x1_512 + xs.max()
                    miny = y1_512 + ys.min(); maxy = y1_512 + ys.max()
                    # 映射回原图
                    scale_x = orig_w / mask512.shape[1]; scale_y = orig_h / mask512.shape[0]
                    nx1 = int(minx * scale_x); nx2 = int(maxx * scale_x)
                    ny1 = int(miny * scale_y); ny2 = int(maxy * scale_y)
                    # 若面积仍过小则外扩固定 margin
                    if (nx2-nx1+1)*(ny2-ny1+1) / img_area < 0.010:
                        margin = 10
                        nx1 = max(0, nx1 - margin); ny1 = max(0, ny1 - margin)
                        nx2 = min(orig_w-1, nx2 + margin); ny2 = min(orig_h-1, ny2 + margin)
                    k_room.update({'bbox': (nx1, ny1, nx2, ny2), 'width': nx2-nx1+1, 'height': ny2-ny1+1, 'pixels': (nx2-nx1+1)*(ny2-ny1+1), 'rebuild': 'seg_merge'})
                    rebuilt_any = True
                    print(f"🍳 [厨房重建A] 重建厨房 bbox=({nx1},{ny1},{nx2},{ny2}) 面积比={(nx2-nx1+1)*(ny2-ny1+1)/img_area:.2%} 原占比={area_ratio:.2%}")
                    new_kitchens.append(k_room)
                if rebuilt_any:
                    room_info['厨房'] = new_kitchens
        except Exception as _kreb_e:
            print(f"⚠️ [厨房重建A] 异常: {_kreb_e}")
        # 对于没有OCR检测到的房间，尝试从分割结果中提取
        label_mapping = {v: k for k, v in room_label_mapping.items()}

        for label, room_type in label_mapping.items():
            if len(room_info[room_type]) == 0:  # OCR没有检测到
                mask = enhanced_resized == label
                pixels = np.sum(mask)
                if pixels <= 0:
                    continue

                total_pixels = enhanced_resized.shape[0] * enhanced_resized.shape[1]
                area_ratio = pixels / total_pixels
                max_area_without_ocr = 0.05  # 全局限定 5%

                # 阳台特殊：必须触到图像边界(假设阳台常贴外墙)且面积 <3% 才允许无OCR回退
                if room_type == "阳台":
                    # 边界接触检测
                    border_touch = False
                    ys, xs = np.where(mask)
                    if len(xs) > 0:
                        if (xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1]-1 or ys.max() == mask.shape[0]-1):
                            border_touch = True
                    if not border_touch:
                        print("🚫 [回退过滤] 无OCR阳台未触及边界 -> 丢弃")
                        continue
                    if area_ratio > 0.03:
                        print(f"🚫 [回退过滤] 无OCR阳台面积过大 {area_ratio:.1%} > 3.0% -> 丢弃")
                        continue

                if area_ratio > max_area_without_ocr:
                    print(f"⚠️ [第3层-融合决策器] 跳过过大的无OCR支持{room_type}区域: {area_ratio:.1%} > {max_area_without_ocr:.1%}")
                    continue

                # 找到房间区域的坐标
                ys, xs = np.where(mask)
                min_x_512, max_x_512 = xs.min(), xs.max()
                min_y_512, max_y_512 = ys.min(), ys.max()
                center_x_512 = int(xs.mean())
                center_y_512 = int(ys.mean())

                scale_x = original_width / 512.0
                scale_y = original_height / 512.0
                center_x = int(center_x_512 * scale_x)
                center_y = int(center_y_512 * scale_y)
                min_x = int(min_x_512 * scale_x)
                max_x = int(max_x_512 * scale_x)
                min_y = int(min_y_512 * scale_y)
                max_y = int(max_y_512 * scale_y)
                width = max_x - min_x + 1
                height = max_y - min_y + 1

                room_info[room_type].append({
                    "center": (center_x, center_y),
                    "bbox": (min_x, min_y, max_x, max_y),
                    "pixels": int(pixels),
                    "width": width,
                    "height": height,
                    "text": "分割检测",
                    "raw_text": "",
                    "confidence": 0.35,
                    "source": "segmentation_fallback"
                })
                print(f"ℹ️ [回退添加] {room_type} (无OCR) bbox=({min_x},{min_y},{max_x},{max_y}) 面积比={area_ratio:.2%}")

        # 合并相近的同类型房间（如中英文标识的同一房间）
        room_info = self._merge_nearby_rooms(room_info, original_size)
        
        # 🔒 严格过滤书房：只有OCR验证的书房才能保留
        if "书房" in room_info:
            ocr_verified_study_rooms = []
            for room in room_info["书房"]:
                # 只保留有OCR文字验证的书房（不是"分割检测"）
                if room.get("text", "") != "分割检测":
                    ocr_verified_study_rooms.append(room)
                    print(f"✅ [书房验证] 保留OCR验证的书房: '{room['text']}'")
                else:
                    print(f"🚫 [书房过滤] 移除无OCR支持的AI分割书房区域")
            
            room_info["书房"] = ocr_verified_study_rooms
            if len(ocr_verified_study_rooms) == 0:
                print("📋 [书房过滤] 无OCR验证的书房，最终结果不包含书房")

        # ===== 冲突解析 & 规范化 =====
        try:
            # 1) 规范化卧室误识别标签: 卧空 / 网房 / 卧房 统一展示为 卧室
            if '卧室' in room_info:
                for b in room_info['卧室']:
                    raw_txt = b.get('text','')
                    if any(tok in raw_txt for tok in ['卧空','网房','卧房']):
                        if raw_txt != '卧室':
                            print(f"🔧 [卧室规范化] '{raw_txt}' -> '卧室'")
                        b['text'] = '卧室'

            # 2) 网房 与 厨房 冲突: 若同一位置既出现 厨房 又出现 '网房'(疑似'厨房'被误分), 且高度重叠, 归并为厨房
            if '厨房' in room_info and '卧室' in room_info and room_info['厨房'] and room_info['卧室']:
                def _bbox_iou(a,b):
                    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
                    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
                    if ix2<ix1 or iy2<iy1: return 0.0
                    inter=(ix2-ix1+1)*(iy2-iy1+1)
                    aarea=(ax2-ax1+1)*(ay2-ay1+1); barea=(bx2-bx1+1)*(by2-by1+1)
                    return inter/float(aarea+barea-inter)
                updated_bedrooms=[]
                for b in room_info['卧室']:
                    raw_txt=b.get('raw_text', b.get('text',''))
                    if '网房' not in raw_txt:
                        updated_bedrooms.append(b)
                        continue
                    # 检查与厨房的IoU
                    merged_into_k=False
                    for k in room_info['厨房']:
                        iou=_bbox_iou(b['bbox'], k['bbox'])
                        if iou>0.45:
                            # 合并: 扩大厨房bbox为并集
                            kx1,ky1,kx2,ky2=k['bbox']; bx1,by1,bx2,by2=b['bbox']
                            union_bbox=(min(kx1,bx1), min(ky1,by1), max(kx2,bx2), max(ky2,by2))
                            if union_bbox!=k['bbox']:
                                print(f"🔄 [冲突解析] '网房' 与 '厨房' IoU={iou:.2f} -> 归并并更新厨房bbox")
                                k['bbox']=union_bbox
                                k['width']=union_bbox[2]-union_bbox[0]+1
                                k['height']=union_bbox[3]-union_bbox[1]+1
                                k['pixels']=k['width']*k['height']
                            merged_into_k=True
                            break
                    if not merged_into_k:
                        # IoU不足, 保留为卧室(已规范化 text)
                        updated_bedrooms.append(b)
                room_info['卧室']=updated_bedrooms

            # 3) 客厅越界裁剪: 若客厅 bbox 含有多个其它房间中心点则视为过度扩张, 进行边界回缩
            if '客厅' in room_info and room_info['客厅']:
                other_types=['厨房','卫生间','卧室','书房','阳台']
                for lr in room_info['客厅']:
                    lx1,ly1,lx2,ly2=lr['bbox']
                    # 收集被包含的其它房间中心
                    contained=[]
                    blockers=[]
                    for ot in other_types:
                        for rr in room_info.get(ot,[]):
                            cx,cy=rr['center']
                            if lx1<=cx<=lx2 and ly1<=cy<=ly2:
                                contained.append((ot, rr))
                                blockers.append(rr['bbox'])
                    if len(contained)>=2:
                        print(f"⚠️ [客厅修正] 客厅包含 {len(contained)} 个其它房间中心 -> 尝试裁剪")
                        # 逐个阻挡框回缩客厅边界
                        for bx1,by1,bx2,by2 in blockers:
                            # 优先沿距离较近的方向收缩
                            # 左侧收缩
                            if bx2< (lx1+lx2)//2 and bx2>lx1 and (bx2-lx1) < (lx2-bx1):
                                lx1 = min(max(lx1, bx2+3), lx2-10)
                            # 右侧收缩
                            if bx1> (lx1+lx2)//2 and bx1<lx2 and (lx2-bx1) < (bx2-lx1):
                                lx2 = max(min(lx2, bx1-3), lx1+10)
                            # 上侧收缩
                            if by2< (ly1+ly2)//2 and by2>ly1 and (by2-ly1) < (ly2-by1):
                                ly1 = min(max(ly1, by2+3), ly2-10)
                            # 下侧收缩
                            if by1> (ly1+ly2)//2 and by1<ly2 and (ly2-by1) < (by2-ly1):
                                ly2 = max(min(ly2, by1-3), ly1+10)
                        # 更新
                        new_bbox=(lx1,ly1,lx2,ly2)
                        if new_bbox!=lr['bbox']:
                            lr['bbox']=new_bbox
                            lr['width']=lx2-lx1+1
                            lr['height']=ly2-ly1+1
                            lr['pixels']=lr['width']*lr['height']
                            print(f"✅ [客厅修正] 裁剪后bbox={new_bbox}")
        except Exception as _conf_e:
            print(f"⚠️ [冲突/越界处理异常] {_conf_e}")
        
        return room_info
        
    def _merge_nearby_rooms(self, room_info, original_size):
        """合并距离很近的同类型房间"""
        print("🔄 检查并合并相近的同类型房间...")
        # 基础合并距离阈值（像素）
        base_merge_threshold = 50
        # 卧室更严格，避免将多个卧室合并成一个
        bedroom_merge_threshold = 35
        # 需要同时满足中心距离阈值 AND 边界框 IoU >= 0.35 或 一方 bbox 完全包含另一方
        
        merged_room_info = {}
        
        for room_type, room_list in room_info.items():
            if len(room_list) <= 1:
                # 只有0或1个房间，无需合并
                merged_room_info[room_type] = room_list
                continue
                
            # 对有多个房间的类型进行合并检查
            merged_list = []
            processed = set()
            
            for i, room1 in enumerate(room_list):
                if i in processed:
                    continue
                    
                # 寻找与当前房间距离很近的其他房间
                to_merge = [room1]
                processed.add(i)
                
                for j, room2 in enumerate(room_list[i+1:], i+1):
                    if j in processed:
                        continue
                        
                    # 计算两房间中心点距离
                    x1, y1 = room1['center']
                    x2, y2 = room2['center']
                    distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    
                    # 计算 bbox 重叠情况
                    bx11,by11,bx12,by12 = room1['bbox']
                    bx21,by21,bx22,by22 = room2['bbox']
                    inter_x1 = max(bx11,bx21); inter_y1 = max(by11,by21)
                    inter_x2 = min(bx12,bx22); inter_y2 = min(by12,by22)
                    inter_area = 0
                    if inter_x2>=inter_x1 and inter_y2>=inter_y1:
                        inter_area = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
                    area1 = (bx12-bx11+1)*(by12-by11+1)
                    area2 = (bx22-bx21+1)*(by22-by21+1)
                    union_area = area1+area2-inter_area if (area1+area2-inter_area)>0 else 1
                    iou = inter_area/float(union_area)
                    contains = (inter_area==area1) or (inter_area==area2)
                    thr = bedroom_merge_threshold if room_type=="卧室" else base_merge_threshold
                    if distance < thr and (iou>=0.35 or contains):
                        to_merge.append(room2); processed.add(j)
                        print(f"   🔗 {room_type}合并：'{room1['text']}' + '{room2['text']}' 距离={distance:.1f} IoU={iou:.2f} contains={contains}")
                    else:
                        if room_type=="卧室" and distance < thr:
                            print(f"   🚫 卧室保持分离：距离{distance:.1f}<阈值{thr}但 IoU={iou:.2f} 且不包含 -> 视为多卧")
                
                if len(to_merge) > 1:
                    # 需要合并多个房间
                    merged_room = self._merge_room_group(to_merge)
                    merged_list.append(merged_room)
                    print(f"   ✅ {room_type}合并完成：{len(to_merge)}个区域 -> 1个区域")
                else:
                    # 单个房间，直接添加
                    merged_list.append(room1)
            
            merged_room_info[room_type] = merged_list
            
            # 输出合并结果
            if len(room_list) != len(merged_list):
                print(f"   📊 {room_type}：{len(room_list)}个 -> {len(merged_list)}个")
        
        return merged_room_info
    
    def _merge_room_group(self, room_group):
        """将多个房间合并为一个"""
        # 选择置信度最高的房间作为基准
        best_room = max(room_group, key=lambda r: r['confidence'])
        
        # 计算合并后的中心点（加权平均，置信度作为权重）
        total_weight = sum(r['confidence'] for r in room_group)
        if total_weight > 0:
            weighted_x = sum(r['center'][0] * r['confidence'] for r in room_group) / total_weight
            weighted_y = sum(r['center'][1] * r['confidence'] for r in room_group) / total_weight
            merged_center = (int(weighted_x), int(weighted_y))
        else:
            # 如果权重为0，使用简单平均
            avg_x = sum(r['center'][0] for r in room_group) / len(room_group)
            avg_y = sum(r['center'][1] for r in room_group) / len(room_group)
            merged_center = (int(avg_x), int(avg_y))
        
        # 计算合并后的边界框（包含所有房间的边界）
        all_bbox = [r['bbox'] for r in room_group]
        min_x = min(bbox[0] for bbox in all_bbox)
        min_y = min(bbox[1] for bbox in all_bbox)
        max_x = max(bbox[2] for bbox in all_bbox)
        max_y = max(bbox[3] for bbox in all_bbox)
        
        merged_width = max_x - min_x + 1
        merged_height = max_y - min_y + 1
        
        # 合并文本描述
        texts = [r['text'] for r in room_group]
        merged_text = ' + '.join(texts)
        
        # 使用最高置信度
        max_confidence = max(r['confidence'] for r in room_group)
        
        return {
            'center': merged_center,
            'bbox': (min_x, min_y, max_x, max_y),
            'pixels': merged_width * merged_height,
            'width': merged_width,
            'height': merged_height,
            'text': merged_text,
            'confidence': max_confidence,
        }

        return room_info

    def _print_room_coordinates(self, room_info, original_size):
        """打印房间坐标详细信息，支持多个同类型房间"""
        print("\n" + "=" * 60)
        print("📍 房间坐标详细信息")
        print("=" * 60)
        print(f"📏 图像尺寸: {original_size[0]} x {original_size[1]} (宽 x 高)")
        print("-" * 60)

        total_rooms = 0
        for room_type, room_list in room_info.items():
            if len(room_list) > 0:
                for i, info in enumerate(room_list):
                    if info["pixels"] > 0:
                        center_x, center_y = info["center"]
                        min_x, min_y, max_x, max_y = info["bbox"]

                        # 如果有多个同类型房间，显示编号
                        if len(room_list) > 1:
                            display_name = f"{room_type}{i+1}"
                        else:
                            display_name = room_type

                        print(f"🏠 {display_name}:")
                        print(f"   📍 中心坐标: ({center_x}, {center_y})")
                        print(
                            f"   📐 边界框: 左上({min_x}, {min_y}) -> 右下({max_x}, {max_y})"
                        )
                        print(f"   📏 尺寸: {info['width']} x {info['height']} 像素")
                        print(f"   📊 面积: {info['pixels']} 像素")
                        print(
                            f"   📄 识别文本: '{info['text']}' (置信度: {info['confidence']:.3f})"
                        )
                        print(f"   🔗 坐标范围: X[{min_x}-{max_x}], Y[{min_y}-{max_y}]")
                        print("-" * 60)
                        total_rooms += 1

            # 如果该类型房间未检测到
            if len(room_list) == 0:
                print(f"❌ {room_type}: 未检测到")
                print("-" * 60)

        print("💡 坐标系说明:")
        print("   • 原点(0,0)在图像左上角")
        print("   • X轴向右为正方向")
        print("   • Y轴向下为正方向")
        print("   • 所有坐标单位为像素")
        print("=" * 60)
        print(f"\n📊 总计检测到 {total_rooms} 个房间")
        print("=" * 60)

    def process(self, image_path, output_path=None):
        """完整处理流程 - 使用四层智能决策架构"""
        try:
            # 设置输出路径
            if output_path is None:
                output_path = Path(image_path).stem

            # 1. 加载模型
            self.load_model()

            # 2. 图像预处理
            img_array, original_size, original_img = self.preprocess_image(image_path)

            # 3-6. 使用四层智能决策架构处理
            results = self.process_with_four_layer_architecture(
                img_array, original_img, original_size
            )

            # 7. 生成结果
            standard_result_path = self.generate_results(
                results['ai_raw'],
                results['ocr_results'],
                results['fusion_result'],
                results['final_result'],
                original_img,
                original_size,
                output_path,
                results['ocr_results']
            )

            # 8. 显示摘要
            self._print_summary()

            return standard_result_path

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            raise

    def _print_summary(self):
        """打印检测摘要 - 基于OCR验证的真实房间数量"""
        # 房间标签到名称/图标/颜色的映射
        label_info = {
            7: ("厨房", "🍳", "绿色"),
            2: ("卫生间", "🚿", "蓝色"),
            3: ("客厅", "🏠", "橙色"),
            4: ("卧室", "🛏️", "紫色"),
            6: ("阳台", "🌞", "青色"),
            8: ("书房", "📚", "棕色"),
        }

        # 基于实际检测到的房间坐标信息统计（避免AI分割误识别）
        if hasattr(self, 'last_room_info'):
            room_counts = {}
            for label, (name, _, _) in label_info.items():
                # 统计实际验证过的房间数量，而不是AI分割的连通域数量
                room_type_map = {7: "厨房", 2: "卫生间", 3: "客厅", 4: "卧室", 6: "阳台", 8: "书房"}
                room_type = room_type_map.get(label, "")
                if room_type and room_type in self.last_room_info:
                    # 只统计有效检测的房间（像素>0的房间）
                    valid_rooms = [r for r in self.last_room_info[room_type] if r.get('pixels', 0) > 0]
                    room_counts[label] = len(valid_rooms)
                else:
                    room_counts[label] = 0
        else:
            # 回退到原来的连通域统计方法（但添加警告）
            print("⚠️ 警告：使用AI分割结果统计，可能包含误识别")
            room_counts = {}
            for label, (name, _, _) in label_info.items():
                mask = self.last_enhanced == label
                pixel_count = np.count_nonzero(mask)
                if pixel_count > 0:
                    num, _ = cv2.connectedComponents(mask.astype(np.uint8))
                    count = num - 1  # 去除背景
                else:
                    count = 0
                room_counts[label] = count

        total_rooms = sum(room_counts.values())
        summary_parts = [
            f"{room_counts[label]}个{name}"
            for label, (name, _, _) in label_info.items()
        ]
        print(f"\n🏠 检测摘要（基于OCR验证）: {' + '.join(summary_parts)} = {total_rooms}个房间")

        # 输出存在的房间类型及其颜色说明
        for label, (name, emoji, color) in label_info.items():
            if room_counts[label] > 0:
                print(f"{emoji} {name}检测: {color}标记")
                
        # 如果发现AI分割与OCR验证结果不一致，提供额外说明
        ai_room_counts = {}
        for label, (name, _, _) in label_info.items():
            mask = self.last_enhanced == label
            pixel_count = np.count_nonzero(mask)
            if pixel_count > 0:
                num, _ = cv2.connectedComponents(mask.astype(np.uint8))
                ai_count = num - 1
            else:
                ai_count = 0
            ai_room_counts[label] = ai_count
            
        # 检查是否有差异并提醒
        for label, (name, _, _) in label_info.items():
            if ai_room_counts[label] != room_counts[label]:
                print(f"⚠️ 注意：{name}的AI分割检测到{ai_room_counts[label]}个区域，但OCR验证后确认为{room_counts[label]}个")
                if ai_room_counts[label] > room_counts[label]:
                    print(f"   💡 可能存在AI误识别，建议查看图像中是否有{name}的错误蓝色标记")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'ai_engine') and hasattr(self.ai_engine, 'session') and self.ai_engine.session:
            self.ai_engine.session.close()

    @staticmethod
    def open_image_with_system_viewer(image_path):
        """使用系统默认图片查看器打开图片"""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(image_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            elif system == "Linux":
                subprocess.run(["xdg-open", image_path])
            else:
                print(f"⚠️ 不支持的操作系统: {system}，无法自动打开图片")
                return False
            
            print(f"🖼️ 已使用系统默认查看器打开: {image_path}")
            return True
        except Exception as e:
            print(f"❌ 打开图片失败: {e}")
            print(f"📂 请手动查看结果: {image_path}")
            return False


def open_image_with_photos_app(image_path):
    """使用Windows照片应用打开图片"""
    try:
        system = platform.system()
        if system == "Windows":
            # 获取绝对路径
            abs_path = os.path.abspath(image_path)
            print(f"🔍 尝试打开文件: {abs_path}")
            
            # 检查文件是否存在
            if not os.path.exists(abs_path):
                print(f"❌ 文件不存在: {abs_path}")
                return False
            
            # 方法1: 使用os.startfile (Windows推荐方式)
            try:
                os.startfile(abs_path)
                print(f"📸 已使用系统默认查看器（通常是照片应用）打开: {os.path.basename(image_path)}")
                return True
            except Exception as e:
                print(f"⚠️ os.startfile失败: {e}")
                
                # 方法2: 尝试使用explorer
                try:
                    subprocess.run(["explorer", abs_path], check=True)
                    print(f"📸 已使用资源管理器打开: {os.path.basename(image_path)}")
                    return True
                except subprocess.CalledProcessError:
                    print("⚠️ explorer方法也失败")
                    
                    # 方法3: 尝试PowerShell Invoke-Item
                    try:
                        subprocess.run([
                            "powershell.exe", 
                            "-Command", 
                            f"Invoke-Item '{abs_path}'"
                        ], check=True)
                        print(f"� 已使用PowerShell打开: {os.path.basename(image_path)}")
                        return True
                    except subprocess.CalledProcessError as e:
                        print(f"❌ PowerShell方法也失败: {e}")
                        return False
        else:
            print(f"⚠️ 照片应用仅支持Windows系统，当前系统: {system}")
            # 回退到系统默认查看器
            return FloorplanProcessor.open_image_with_system_viewer(image_path)
    except Exception as e:
        print(f"❌ 打开图片失败: {e}")
        print(f"📂 请手动查看结果: {image_path}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DeepFloorplan 房间检测 - 重构版本 (带坐标轴)"
    )
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出文件名前缀")
    parser.add_argument("--model", "-m", default="pretrained", help="模型路径")
    parser.add_argument(
        "--fonts",
        help="逗号分隔的字体列表，按优先级使用",
    )

    args = parser.parse_args()

    # 字体已在模块加载时初始化，若未成功提供一次运行期提示
    if CH_FONT is None:
        print("⚠️ 未找到可用中文字体，可能出现 ? 号。可使用 --fonts 指定，但需系统已安装对应字体。")

    # 检查输入文件
    if not Path(args.image).exists():
        print(f"❌ 输入文件不存在: {args.image}")
        sys.exit(1)

    # 创建处理器并执行
    processor = FloorplanProcessor(args.model)
    standard_result_path = processor.process(args.image, args.output)

    # 确定输出文件路径
    output_base = args.output if args.output else Path(args.image).stem
    coordinate_result_path = f"output/{output_base}_coordinate_result.png"
    
    print("\n🎉 处理完成！")
    print("📂 输出目录: output/")
    print("🖼️ 主要结果:")
    print(f"   📊 带坐标轴结果: {coordinate_result_path}")
    print(f"   📸 标准结果: {standard_result_path}")
    
    # 自动打开生成的图片
    print("\n🖼️ 正在打开结果图片...")
    
    # 优先打开带坐标轴的结果图（更详细）
    if os.path.exists(coordinate_result_path):
        if open_image_with_photos_app(coordinate_result_path):
            print("✅ 已使用照片应用打开带坐标轴的结果图")
        else:
            print("⚠️ 无法自动打开图片，请手动查看结果文件")
    elif os.path.exists(standard_result_path):
        if open_image_with_photos_app(standard_result_path):
            print("✅ 已使用照片应用打开标准结果图")
        else:
            print("⚠️ 无法自动打开图片，请手动查看结果文件")
    else:
        print("❌ 找不到生成的结果图片文件")


if __name__ == "__main__":
    main()
