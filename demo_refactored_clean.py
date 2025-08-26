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
import cv2
from PIL import Image

CH_FONT_PATH = None

def _init_chinese_font():
    """初始化中文字体，防止出现 ? 号。返回可用 FontProperties 或 None。"""
    candidate_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun", "Source Han Sans CN",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"
    ]
    for name in candidate_fonts:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path and os.path.isfile(path):
                print(f"🈶 使用中文字体: {name} -> {path}")
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                global CH_FONT_PATH
                CH_FONT_PATH = path
                return FontProperties(fname=path)
        except Exception:
            continue
    print("⚠️ 未找到适配的中文字体，可能出现 ? 号，请安装微软雅黑/黑体。")
    return None

CH_FONT = _init_chinese_font()


tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

# 导入原有工具模块
from utils.ocr_enhanced import extract_room_text, fuse_ocr_and_segmentation, text_to_label
from utils.rgb_ind_convertor import floorplan_fuse_map_figure
from room_detection_manager import RefactoredRoomDetectionManager


# ============================================================
# 四层智能决策架构
# ============================================================

class AISegmentationEngine:
    """第一层：AI语义分割器"""
    
    def __init__(self, model_path="pretrained"):
        self.model_path = model_path
        self.session = None
        self.inputs = None
        self.room_type_logit = None
        self.room_boundary_logit = None
    
    def load_model(self):
        """加载神经网络模型"""
        print("🔧 [第1层-AI分割器] 加载DeepFloorplan模型...")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph(f"{self.model_path}/pretrained_r3d.meta")
        saver.restore(self.session, f"{self.model_path}/pretrained_r3d")

        graph = tf.get_default_graph()
        self.inputs = graph.get_tensor_by_name("inputs:0")
        self.room_type_logit = graph.get_tensor_by_name("Cast:0")
        self.room_boundary_logit = graph.get_tensor_by_name("Cast_1:0")
        print("✅ [第1层-AI分割器] 模型加载完成")
    
    def segment_image(self, img_array):
        """执行语义分割"""
        print("🤖 [第1层-AI分割器] 运行神经网络推理...")
        input_batch = np.expand_dims(img_array, axis=0)

        # 原网络图中 Cast/Cast_1 节点已经输出类别索引，此处无需再次 argmax
        room_type, room_boundary = self.session.run(
            [self.room_type_logit, self.room_boundary_logit],
            feed_dict={self.inputs: input_batch},
        )

        room_type = np.squeeze(room_type)
        room_boundary = np.squeeze(room_boundary)

        # 将边界类别映射到 9/10，供后续融合流程识别墙体
        floorplan = room_type.copy()
        floorplan[room_boundary == 1] = 9
        floorplan[room_boundary == 2] = 10

        print("✅ [第1层-AI分割器] 神经网络推理完成")
        return floorplan


class OCRRecognitionEngine:
    """第二层：OCR文字识别器"""
    
    def __init__(self):
        pass
    
    def recognize_text(self, original_img):
        """识别图像中的文字"""
        print("🔍 [第2层-OCR识别器] 提取OCR文字信息...")
        
        # OCR处理（放大2倍提高识别率）
        ocr_img = cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        print(f"🔍 [第2层-OCR识别器] 处理图像: {ocr_img.shape[1]} x {ocr_img.shape[0]} (放大2倍)")
        
        room_text_items = extract_room_text(ocr_img)
        print(f"📊 [第2层-OCR识别器] 检测到 {len(room_text_items)} 个文字区域")

        # 保存OCR放大图尺寸，便于后续可视化/坐标还原
        for it in room_text_items:
            it['ocr_width'] = ocr_img.shape[1]
            it['ocr_height'] = ocr_img.shape[0]
        
        return room_text_items, ocr_img.shape


class FusionDecisionEngine:
    """第三层：融合决策器"""
    
    def __init__(self):
        self.room_manager = RefactoredRoomDetectionManager()
        # 记录OCR驱动区域扩散的种子点 (label -> [(x,y), ...])
        self._seed_centers_by_label = {}
    
    def fuse_results(self, ai_prediction, ocr_results, ocr_shape):
        """智能融合AI分割和OCR识别结果"""
        print("🔗 [第3层-融合决策器] 融合AI和OCR结果...")

        # 1. 计算 OCR -> 512 缩放比例（当前 OCR 是放大2倍后的尺寸）
        ocr_to_512_x = 512.0 / ocr_shape[1]
        ocr_to_512_y = 512.0 / ocr_shape[0]
        print(f"   🔄 [第3层-融合决策器] OCR坐标转换到512x512:")
        print(f"      OCR图像({ocr_shape[1]}x{ocr_shape[0]}) -> 512x512")
        print(f"      转换比例: X={ocr_to_512_x:.3f}, Y={ocr_to_512_y:.3f}")

        # 2. 复制原始OCR结果（保持放大图坐标，后续需要用比例映射）
        original_ocr_results = [item.copy() for item in ocr_results]
        for item in original_ocr_results:
            item['ocr_width'] = ocr_shape[1]
            item['ocr_height'] = ocr_shape[0]

        # 3. 生成 512 坐标系版本供直接融合与房间检测
        converted_items = self._convert_ocr_coordinates(original_ocr_results, ocr_to_512_x, ocr_to_512_y)

        # 4. 识别开放式厨房（厨房文字落在客厅区域）
        processed_items = []
        open_kitchens = []
        for item in converted_items:
            label = text_to_label(item['text'])
            if label == 7:  # 厨房
                x, y, w, h = item['bbox']
                cx, cy = x + w // 2, y + h // 2
                if ai_prediction[cy, cx] == 3:  # 客厅标签
                    open_kitchens.append(item)
                    print(f"   🍳 识别到开放式厨房候选: {item['text']}")
                else:
                    processed_items.append(item)
            else:
                processed_items.append(item)

        # 5. 融合 OCR 标签（不含开放式厨房）
        enhanced = fuse_ocr_and_segmentation(ai_prediction.copy(), processed_items)

        # 6. 开放式厨房区域估算
        enhanced = self._estimate_open_kitchen(enhanced, open_kitchens)

        # 7. OCR 主导区域扩散（使用原始坐标 + 比例）
        enhanced = self._ocr_driven_region_growing(enhanced, original_ocr_results, ocr_to_512_x, ocr_to_512_y)

        # 8. 房间检测（使用已缩放的 converted_items，避免再 clamp 511）
        enhanced = self.room_manager.detect_all_rooms(enhanced, converted_items)

        # 9. 基础清理（距离计算仍需原始坐标 + 比例）
        enhanced = self._basic_cleanup(enhanced, original_ocr_results, ocr_to_512_x, ocr_to_512_y)

        print("✅ [第3层-融合决策器] 融合完成")
        return enhanced
    
    def _convert_ocr_coordinates(self, room_text_items, scale_x, scale_y):
        """转换OCR坐标到512x512坐标系"""
        converted_items = []
        for item in room_text_items:
            converted_item = item.copy()
            x, y, w, h = item["bbox"]
            new_x = max(0, min(int(x * scale_x), 511))
            new_y = max(0, min(int(y * scale_y), 511))
            new_w = max(1, min(int(w * scale_x), 512 - new_x))
            new_h = max(1, min(int(h * scale_y), 512 - new_y))
            converted_item["bbox"] = [new_x, new_y, new_w, new_h]
            converted_items.append(converted_item)
        return converted_items

    def _estimate_open_kitchen(self, enhanced, kitchen_items, size=60):
        """开放式厨房区域估算：当厨房文字落在客厅区域中时估计其范围"""
        if not kitchen_items:
            return enhanced
        print("🍳 [第3层-融合决策器] 估算开放式厨房区域...")
        for item in kitchen_items:
            x, y, w, h = item['bbox']
            cx, cy = x + w // 2, y + h // 2
            half = size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(enhanced.shape[1] - 1, cx + half)
            y2 = min(enhanced.shape[0] - 1, cy + half)
            print(f"   ➕ 开放式厨房区域: ({x1}, {y1}) -> ({x2}, {y2})")
            patch = enhanced[y1:y2, x1:x2]
            mask = ~np.isin(patch, [9, 10])  # 避开墙体
            patch[mask] = 7
        return enhanced
    
    def _ocr_driven_region_growing(self, enhanced, original_ocr_results, scale_x, scale_y):
        """OCR主导的区域生长算法 - 从OCR位置向外扩散至边界并上色"""
        print("🌱 [第3层-融合决策器] OCR主导区域扩散...")
        
        # 处理每个OCR检测到的房间文字
        for item in original_ocr_results:
            text = item["text"].lower().strip()
            confidence = item.get("confidence", 1.0)
            
            # 确定房间类型
            room_label = None
            room_name = ""
            
            if any(keyword in text for keyword in ["厨房", "kitchen", "厨"]):
                room_label = 7  # 厨房
                room_name = "厨房"
            elif any(keyword in text for keyword in ["卫生间", "bathroom", "卫", "洗手间", "浴室"]):
                room_label = 2  # 卫生间  
                room_name = "卫生间"
            elif any(keyword in text for keyword in ["客厅", "living", "厅", "起居室"]):
                room_label = 3  # 客厅
                room_name = "客厅"
            elif any(keyword in text for keyword in ["卧室", "bedroom", "主卧", "次卧"]):
                room_label = 4  # 卧室
                room_name = "卧室"
                print(f"🔍 [调试] OCR检测到卧室关键词: '{text}' -> 卧室(4)")
            elif any(keyword in text for keyword in ["书房", "study", "办公室", "office"]):
                room_label = 8  # 书房
                room_name = "书房"
                print(f"🔍 [调试] OCR检测到书房关键词: '{text}' -> 书房(8)")
            elif any(keyword in text for keyword in ["阳台", "balcony", "阳兮", "阳合", "阳囊"]):
                room_label = 6  # 阳台
                room_name = "阳台"
                if text == "阳兮":
                    print(f"🔧 [OCR修正] 误识别'{text}' -> '阳台'")
            
            if room_label is None:
                continue
                
            print(f"   🎯 处理房间: '{text}' -> {room_name}({room_label}) (置信度: {confidence:.3f})")
            
            # 转换OCR坐标到512x512坐标系
            x, y, w, h = item["bbox"]
            center_x_512 = int((x + w//2) * scale_x)
            center_y_512 = int((y + h//2) * scale_y)

            # 计算并裁剪OCR框在512坐标系下的范围
            x1_512 = max(0, min(int(x * scale_x), 511))
            y1_512 = max(0, min(int(y * scale_y), 511))
            x2_512 = max(0, min(int((x + w) * scale_x), 512))
            y2_512 = max(0, min(int((y + h) * scale_y), 512))

            # 确保中心坐标在有效范围内
            center_x_512 = max(0, min(center_x_512, 511))
            center_y_512 = max(0, min(center_y_512, 511))

            # 从OCR位置开始区域生长
            room_mask = self._region_growing_from_seed(
                enhanced, center_x_512, center_y_512, room_label, (x1_512, y1_512, x2_512, y2_512)
            )

            # 记录种子点，供后续清理阶段判定主区域
            self._seed_centers_by_label.setdefault(room_label, []).append((center_x_512, center_y_512))
            
            if room_mask is not None:
                room_pixels = np.sum(room_mask)
                print(f"   ✅ {room_name}区域扩散完成: {room_pixels} 像素，中心({center_x_512}, {center_y_512})")
                
                # 应用区域生长结果
                enhanced[room_mask] = room_label
            else:
                print(f"   ⚠️ {room_name}区域扩散失败，使用备用方法")
                # 备用方法：创建小的固定区域
                self._create_fallback_room_region(enhanced, center_x_512, center_y_512, room_label, room_name)
        
        return enhanced
    
    def _region_growing_from_seed(self, floorplan, seed_x, seed_y, target_label, bbox=None):
        """从种子点开始区域生长，直到遇到边界（墙体或其他房间）"""
        h, w = floorplan.shape
        
        # 检查种子点是否有效
        if (seed_x < 0 or seed_x >= w or seed_y < 0 or seed_y >= h):
            return None
            
        # 如果种子点在墙上，尝试寻找附近的非墙区域
        if floorplan[seed_y, seed_x] in [9, 10]:  # 墙体
            seed_x, seed_y = self._find_nearby_non_wall(floorplan, seed_x, seed_y, bbox)
            if seed_x is None:
                print("      ❌ 无法在附近找到非墙像素，区域扩散终止")
                return None
        
        print(f"      🌱 开始从种子点({seed_x}, {seed_y})扩散，初始值: {floorplan[seed_y, seed_x]}")
        
        # 区域生长算法（BFS）
        from collections import deque
        
        visited = np.zeros((h, w), dtype=bool)
        room_mask = np.zeros((h, w), dtype=bool)
        queue = deque([(seed_x, seed_y)])
        
        # 🎯 严格边界策略：避开墙体和已有房间
        wall_barriers = {9, 10}  # 墙体
        room_barriers = {2, 3, 4, 6, 7, 8}  # 其他房间类型
        
        expand_count = 0
        # 根据图像大小动态确定最大扩散次数，进一步放宽房间扩散限制
        total_pixels = h * w
        # 🎯 优化扩散限制：按房间类型设置不同的扩散系数
        expansion_factor = {
            2: 0.7,   # 卫生间需要适度扩散
            3: 0.8,   # 客厅需要大范围扩散
            4: 0.75,  # 卧室需要中等扩散
            6: 0.75,  # 阳台提升扩散（之前可能不足）
            7: 0.7,   # 厨房适度扩散
            8: 0.5,   # 书房控制扩散（防止误识别）
        }.get(target_label, 0.6)
        expansion_limit = int(total_pixels * expansion_factor)
        encountered_wall = False

        # 🔒 边界检测：适度的安全边距
        safe_margin = 2  # 减少到2像素的安全边距

        while queue:
            x, y = queue.popleft()
            expand_count += 1
            
            # 🚫 适度边界检查：包括图像边界和安全边距
            if (x < safe_margin or x >= w - safe_margin or 
                y < safe_margin or y >= h - safe_margin or 
                visited[y, x]):
                continue
                
            visited[y, x] = True
            current_pixel = floorplan[y, x]
            
            # 🚫 绝对边界：墙体 - 绝不越过
            if current_pixel in wall_barriers:
                encountered_wall = True
                continue

            # 🤔 智能边界判断：避免覆盖其他已确定的房间（带小组件宽容）
            if current_pixel in room_barriers and current_pixel != target_label:
                distance_to_seed = max(abs(x - seed_x), abs(y - seed_y))
                max_override_distance = {
                    2: 15,  # 卫生间允许15像素覆盖
                    3: 25,  # 客厅允许25像素覆盖
                    4: 20,  # 卧室允许20像素覆盖
                    6: 12,  # 阳台允许12像素覆盖
                    7: 18,  # 厨房允许18像素覆盖
                    8: 20,  # 书房允许20像素覆盖
                }.get(target_label, 15)
                small_area_thresh = 30
                near_seed_thresh = 5
                component_area = self._compute_component_area(floorplan, x, y, current_pixel)
                if (component_area >= small_area_thresh and
                        distance_to_seed > near_seed_thresh and
                        distance_to_seed > max_override_distance):
                    continue  # 被较大组件阻挡且距离较远，停止覆盖
            
            # 添加到房间掩码
            room_mask[y, x] = True
            
            # 🎯 针对客厅优化：多方向均匀扩散
            if target_label == 3:  # 客厅
                # 8方向扩散（包括对角线），确保全方向覆盖
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    queue.append((x + dx, y + dy))
            else:
                # 其他房间4方向扩散（避免对角线过度扩散）
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((x + dx, y + dy))
        
            if expand_count >= expansion_limit:
                if queue and not encountered_wall:
                    expansion_limit += int(total_pixels * 0.05)
                else:
                    print(f"      ⚠️ 达到扩散限制({expansion_limit})，停止扩散")
                    break
        
        # 区域生长完成后，进行闭运算去除噪点并拟合规整形状
        room_mask = self._refine_room_mask(room_mask)

        # 检查生成的区域是否合理
        room_pixels = np.sum(room_mask)
        room_ratio = room_pixels / total_pixels

        # 🎯 根据墙体检测动态调整最大面积比例
        wall_area = np.sum(np.isin(floorplan, list(wall_barriers)))
        building_area = max(total_pixels - wall_area, 1)
        base_max_ratio = {
            2: 0.30,  # 卫生间最多30%
            3: 0.70,  # 客厅最多70%
            4: 0.50,  # 卧室最多50%
            6: 0.28,  # 阳台放宽到28%
            7: 0.35,  # 厨房最多35%
            8: 0.25,  # 书房最多25%
        }.get(target_label, 0.50)
        max_ratio = base_max_ratio * (building_area / total_pixels)
        
        min_pixels = {
            2: 150,   # 卫生间最少150像素
            3: 300,   # 客厅最少300像素
            4: 200,   # 卧室最少200像素
            6: 60,    # 阳台降低最小像素门槛
            7: 150,   # 厨房最少150像素
            8: 200,   # 书房最少200像素
        }.get(target_label, 150)
        
        if room_ratio > max_ratio:  # 超过房间最大比例
            print(f"      ⚠️ 扩散区域过大({room_ratio:.1%} > {max_ratio:.1%})，进行裁剪")
            room_mask = self._clip_oversized_region(room_mask, floorplan, seed_x, seed_y, target_label)
            room_mask = self._refine_room_mask(room_mask)
            return room_mask
        elif room_pixels < min_pixels:  # 太小也不合理
            print(f"      ⚠️ 扩散区域过小({room_pixels}像素 < {min_pixels}像素)")
            return None
        
        return room_mask
    
    def _find_nearby_non_wall(self, floorplan, center_x, center_y, bbox=None):
        """寻找附近的非墙区域"""
        h, w = floorplan.shape

        # 对墙体进行膨胀，减少墙体小缺口的影响
        wall_mask = np.isin(floorplan, [9, 10]).astype(np.uint8)
        dilated_walls = cv2.dilate(wall_mask, np.ones((3, 3), np.uint8), iterations=1)

        # 在更大范围内搜索可用起点，半径扩大到10-15
        for radius in range(10, 16):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = center_x + dx, center_y + dy
                    if 0 <= nx < w and 0 <= ny < h and not dilated_walls[ny, nx]:
                        return nx, ny

        # 若仍未找到，且提供了OCR框，则在框内细致搜索
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best_pt = None
            best_dist = None
            for ny in range(y1, y2):
                for nx in range(x1, x2):
                    if not dilated_walls[ny, nx]:
                        dist = (nx - cx) ** 2 + (ny - cy) ** 2
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_pt = (nx, ny)
            if best_pt is not None:
                print(f"      🔍 在OCR框内找到替代起点: {best_pt}")
                return best_pt

        print("      ⚠️ 未找到可用的非墙起点")
        return None, None

    def _compute_component_area(self, floorplan, start_x, start_y, label, max_check=100):
        """计算从指定像素开始的连通区域面积，用于判断小组件"""
        from collections import deque
        h, w = floorplan.shape
        visited = set()
        q = deque([(start_x, start_y)])
        area = 0
        while q and area <= max_check:
            x, y = q.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if floorplan[y, x] != label:
                continue
            area += 1
            q.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
        return area

    def _refine_room_mask(self, room_mask):
        """对房间掩码做闭运算并拟合多边形，使形状更规整"""
        mask_uint8 = (room_mask.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined = np.zeros_like(closed)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            eps = 0.01 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, eps, True)
            cv2.fillPoly(refined, [approx], 255)
        else:
            refined = closed

        return refined.astype(bool)
    
    def _clip_oversized_region(self, room_mask, floorplan, seed_x, seed_y, target_label):
        """裁剪过大的区域，利用凸包/最小外接矩形并参考墙体信息"""
        mask_uint8 = (room_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return room_mask

        # 使用最大轮廓计算最小外接矩形和凸包
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)

        hull_mask = np.zeros_like(mask_uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        box_mask = np.zeros_like(mask_uint8)
        cv2.fillPoly(box_mask, [box], 255)

        candidate = cv2.bitwise_and(hull_mask, box_mask)

        # 参考墙体：避免穿过墙体
        if floorplan is not None:
            non_wall = (~np.isin(floorplan, [9, 10])).astype(np.uint8) * 255
            candidate = cv2.bitwise_and(candidate, non_wall)

        clipped = np.logical_and(room_mask, candidate.astype(bool))

        # 若裁剪后不包含种子点，则保留种子附近的小区域
        if not clipped[seed_y, seed_x]:
            circle = np.zeros_like(room_mask, dtype=np.uint8)
            cv2.circle(circle, (seed_x, seed_y), 20, 1, -1)
            clipped = np.logical_or(clipped, circle.astype(bool))

        return clipped
    
    def _create_fallback_room_region(self, enhanced, center_x, center_y, room_label, room_name):
        """创建备用的固定大小房间区域"""
        h, w = enhanced.shape
        
        # 根据房间类型确定大小
        room_size = {
            2: 35,   # 卫生间较小
            3: 70,   # 客厅较大
            4: 55,   # 卧室中等
            7: 45,   # 厨房中等
            8: 50,   # 书房中等
        }.get(room_label, 40)
        
        half_size = room_size // 2
        
        x1 = max(0, center_x - half_size)
        x2 = min(w - 1, center_x + half_size)  
        y1 = max(0, center_y - half_size)
        y2 = min(h - 1, center_y + half_size)
        
        # 只在非墙区域设置房间标签
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if enhanced[y, x] not in [9, 10]:  # 非墙体
                    enhanced[y, x] = room_label
        
        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        print(f"      ✅ 创建备用{room_name}区域: {area} 像素")
    
    def _basic_cleanup(self, enhanced, original_ocr_results, scale_x, scale_y):
        """基础清理：距离阈值清理"""
        print("🧹 [第3层-融合决策器] 基础清理...")
        
        # 获取OCR验证的房间位置（使用原始坐标转换到512x512）
        ocr_rooms = self._extract_ocr_rooms_for_cleanup(original_ocr_results, scale_x, scale_y)
        
        # ⚠️ 跳过卫生间清理，保留OCR扩散结果
        # 清理误识别区域 - 排除卫生间，保留OCR扩散结果
        for room_label, room_positions in ocr_rooms.items():
            if room_label in [3, 4, 7]:  # 只处理客厅、卧室和厨房，跳过卫生间
                enhanced = self._clean_room_type(enhanced, room_label, room_positions)
        
        return enhanced
    
    def _extract_ocr_rooms_for_cleanup(self, room_text_items, scale_x, scale_y):
        """为清理算法提取OCR验证的房间位置（使用原始坐标转换到512x512）"""
        ocr_rooms = {}
        for item in room_text_items:
            text = item["text"].lower().strip()
            room_type = None
            
            if any(keyword in text for keyword in ["厨房", "kitchen", "厨"]):
                room_type = 7
            elif any(keyword in text for keyword in ["卫生间", "bathroom", "卫", "洗手间", "浴室"]):
                room_type = 2
            elif any(keyword in text for keyword in ["卧室", "bedroom", "主卧", "次卧"]):
                room_type = 4
            elif any(keyword in text for keyword in ["客厅", "living", "客", "大厅"]):
                room_type = 3
            
            if room_type:
                if room_type not in ocr_rooms:
                    ocr_rooms[room_type] = []
                
                # 使用OCR的原始坐标并转换到512x512
                x, y, w, h = item["bbox"]  # 这是原始OCR坐标（2倍放大图像上的）
                
                center_512_x = int((x + w//2) * scale_x)
                center_512_y = int((y + h//2) * scale_y)
                
                # 确保坐标在512x512范围内
                center_512_x = max(0, min(center_512_x, 511))
                center_512_y = max(0, min(center_512_y, 511))
                
                ocr_rooms[room_type].append((center_512_x, center_512_y, item["confidence"]))
                print(f"   🎯 [第3层-融合决策器] {text}({room_type}) OCR位置转换: 原始({x+w//2}, {y+h//2}) -> 512x512({center_512_x}, {center_512_y})")
        
        return ocr_rooms
    
    def _extract_ocr_rooms(self, room_text_items):
        """提取OCR验证的房间位置"""
        ocr_rooms = {}
        for item in room_text_items:
            text = item["text"].lower().strip()
            room_type = None
            
            if any(keyword in text for keyword in ["厨房", "kitchen", "厨"]):
                room_type = 7
            elif any(keyword in text for keyword in ["卫生间", "bathroom", "卫", "洗手间", "浴室"]):
                room_type = 2
            # 可以继续添加其他房间类型...
            
            if room_type:
                if room_type not in ocr_rooms:
                    ocr_rooms[room_type] = []
                
                # 注意：这里的item["bbox"]已经是转换后的512x512坐标系的坐标
                x, y, w, h = item["bbox"]
                center_x = int(x + w//2)
                center_y = int(y + h//2)
                ocr_rooms[room_type].append((center_x, center_y, item["confidence"]))
        
        return ocr_rooms
    
    def _clean_room_type(self, enhanced, room_label, room_positions):
        """清理特定房间类型的误识别（保留包含OCR扩散种子的主区域）"""
        room_names = {2: "卫生间", 3: "客厅", 4: "卧室", 7: "厨房"}
        room_name = room_names.get(room_label, "房间")
        print(f"🧹 [第3层-融合决策器] 清理{room_name}误识别，保留{len(room_positions)}个OCR验证位置")

        mask = (enhanced == room_label).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        if num_labels <= 1:
            return enhanced  # 无需清理

        cleaned_mask = np.zeros_like(mask)
        seed_points = self._seed_centers_by_label.get(room_label, [])
        if seed_points:
            print(f"   🧪 调试: {room_name} 记录扩散种子 {len(seed_points)} 个 -> {seed_points[:5]}{'...' if len(seed_points)>5 else ''}")

        # 预计算种子所属连通域 ID
        seed_component_ids = set()
        h_labels, w_labels = labels_im.shape
        for (sx, sy) in seed_points:
            if 0 <= sx < w_labels and 0 <= sy < h_labels:
                cid = labels_im[sy, sx]
                if cid != 0:
                    seed_component_ids.add(cid)
        if seed_component_ids:
            print(f"   🔐 含种子连通域 IDs: {sorted(seed_component_ids)}")

        for comp_id in range(1, num_labels):
            comp_centroid = centroids[comp_id]
            comp_center_x, comp_center_y = int(comp_centroid[0]), int(comp_centroid[1])
            comp_area = stats[comp_id, cv2.CC_STAT_AREA]

            # 计算到最近OCR中心的距离
            min_distance = float('inf')
            for ocr_x, ocr_y, _ in room_positions:
                d = np.hypot(comp_center_x - ocr_x, comp_center_y - ocr_y)
                if d < min_distance:
                    min_distance = d

            # 阈值策略（放宽，避免误删扩散结果）
            if room_label == 3:           # 客厅
                distance_threshold = 260
                max_area_threshold = 90000
            elif room_label == 4:         # 卧室
                distance_threshold = 200
                max_area_threshold = 50000
            elif room_label == 2:         # 卫生间
                distance_threshold = 220
                max_area_threshold = 30000
            elif room_label == 7:         # 厨房
                distance_threshold = 200
                max_area_threshold = 35000
            else:                         # 其他
                distance_threshold = 180
                max_area_threshold = 40000

            # 强制保留：组件ID含种子
            if comp_id in seed_component_ids:
                cleaned_mask[labels_im == comp_id] = 1
                print(f"   ✅ 保留{room_name}区域(种子组件#{comp_id}): 面积:{comp_area}")
                continue

            # 二次确认：组件内部是否包含任一实际种子像素
            contains_seed = False
            if seed_points and comp_id not in seed_component_ids:
                component_mask = (labels_im == comp_id)
                for (sx, sy) in seed_points:
                    if 0 <= sx < w_labels and 0 <= sy < h_labels and component_mask[sy, sx]:
                        contains_seed = True
                        break
            if contains_seed:
                cleaned_mask[labels_im == comp_id] = 1
                print(f"   ✅ 保留{room_name}区域(含种子像素#{comp_id}): 面积:{comp_area}")
                continue

            if min_distance < distance_threshold and comp_area < max_area_threshold:
                cleaned_mask[labels_im == comp_id] = 1
                print(f"   ✅ 保留{room_name}区域：距OCR:{min_distance:.1f}px, 面积:{comp_area}")
            else:
                print(f"   ❌ 移除{room_name}区域：距OCR:{min_distance:.1f}px, 面积:{comp_area}")

        # 清理与重建
        enhanced[mask == 1] = 0
        enhanced[cleaned_mask == 1] = room_label
        # 兜底：若全部删除但有种子连通域，恢复
        if np.sum(cleaned_mask) == 0 and seed_component_ids:
            print(f"   ⚠️ 兜底触发: {room_name} 所有组件被删但存在种子, 恢复种子连通域")
            for comp_id in seed_component_ids:
                enhanced[labels_im == comp_id] = room_label
        elif np.sum(cleaned_mask) == 0 and seed_points:
            print(f"   ⚠️ 兜底2: {room_name} 无保留组件; 在种子点周围创建最小保护块")
            for (sx, sy) in seed_points:
                x1 = max(0, sx-5); x2 = min(w_labels-1, sx+5)
                y1 = max(0, sy-5); y2 = min(h_labels-1, sy+5)
                enhanced[y1:y2+1, x1:x2+1] = room_label
        kept_pixels = np.sum(enhanced == room_label)
        print(f"   📊 清理后{room_name}总像素: {kept_pixels}")
        return enhanced


class ReasonablenessValidator:
    """第四层：合理性验证器"""
    
    def __init__(self):
        self.spatial_rules = SpatialRuleEngine()
        self.size_constraints = SizeConstraintEngine()
        self.boundary_detector = BuildingBoundaryDetector()
    
    def validate_and_correct(self, fused_results, ocr_results, original_size):
        """验证并修正不合理的识别结果"""
        print("🔍 [第4层-合理性验证器] 开始合理性验证...")
        
        # 1. 空间合理性检查
        validated_results = self.spatial_rules.validate_spatial_logic(fused_results, ocr_results)
        
        # 2. 尺寸约束验证
        validated_results = self.size_constraints.validate_size_constraints(validated_results, original_size)
        
        # 3. 边界范围检查
        validated_results = self.boundary_detector.validate_building_boundary(validated_results, original_size)
        
        print("✅ [第4层-合理性验证器] 合理性验证完成")
        return validated_results


class SpatialRuleEngine:
    """空间逻辑规则引擎"""
    
    def validate_spatial_logic(self, results, ocr_results):
        """验证空间逻辑合理性"""
        print("🧠 [空间规则引擎] 验证空间逻辑...")
        
        # 规则1: 检查卧室内的重复房间标记
        results = self._check_nested_rooms(results, ocr_results)
        
        # 规则2: 检查房间重叠冲突
        results = self._check_room_overlap(results, ocr_results)
        
        # 规则3: 检查厨房位置合理性（不应在客厅中央）
        results = self._check_kitchen_position(results, ocr_results)
        
        return results
    
    def _check_nested_rooms(self, results, ocr_results):
        """检查并清理嵌套房间（如卧室内的额外卧室）"""
        print("   🏠 [空间规则引擎] 检查嵌套房间...")
        
        # 获取OCR标注的房间区域
        ocr_room_regions = {}
        for item in ocr_results:
            text = item["text"].lower().strip()
            if any(keyword in text for keyword in ["卧室", "bedroom"]):
                x, y, w, h = item["bbox"]
                
                # OCR bbox是在2倍放大图像上的，需要转换到512x512坐标系
                # OCR图像尺寸可以从item中获取，或者通过原始图像尺寸计算
                ocr_to_512_scale_x = 512.0 / (item.get('ocr_width', 1158))  # 默认值基于demo1.jpg
                ocr_to_512_scale_y = 512.0 / (item.get('ocr_height', 866))
                
                # 转换坐标到512x512
                x_512 = int(x * ocr_to_512_scale_x)
                y_512 = int(y * ocr_to_512_scale_y)
                w_512 = int(w * ocr_to_512_scale_x)
                h_512 = int(h * ocr_to_512_scale_y)
                
                # 扩大OCR区域范围用于检测（扩大到2倍）
                expanded_region = {
                    'x1': max(0, x_512 - w_512),
                    'y1': max(0, y_512 - h_512), 
                    'x2': min(512, x_512 + w_512 + w_512),
                    'y2': min(512, y_512 + h_512 + h_512),
                    'text': text,
                    'center_x': x_512 + w_512//2,
                    'center_y': y_512 + h_512//2
                }
                ocr_room_regions[text] = expanded_region
                print(f"   📍 [空间规则引擎] OCR房间 '{text}': 中心({expanded_region['center_x']}, {expanded_region['center_y']}), 区域({expanded_region['x1']}, {expanded_region['y1']}) -> ({expanded_region['x2']}, {expanded_region['y2']})")
        
        # 检查每个OCR卧室区域内是否有AI分割的其他卧室
        bedroom_mask = (results == 4).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(bedroom_mask, connectivity=4)
        
        print(f"   🔍 [空间规则引擎] 发现 {num_labels-1} 个AI分割的卧室连通域")
        
        for room_name, region in ocr_room_regions.items():
            print(f"   📍 [空间规则引擎] 检查 '{room_name}' 区域内的嵌套房间...")
            
            # 在OCR区域内查找AI分割的卧室块
            nested_components = []
            for comp_id in range(1, num_labels):
                centroid_x, centroid_y = centroids[comp_id]
                area = stats[comp_id, cv2.CC_STAT_AREA]
                
                print(f"   🔍 [空间规则引擎] AI卧室组件{comp_id}: 中心({centroid_x:.1f}, {centroid_y:.1f}), 面积:{area}")
                
                # 检查质心是否在OCR区域内
                if (region['x1'] <= centroid_x <= region['x2'] and 
                    region['y1'] <= centroid_y <= region['y2']):
                    nested_components.append(comp_id)
                    print(f"   ✅ [空间规则引擎] 组件{comp_id}在 '{room_name}' 区域内")
                else:
                    print(f"   ❌ [空间规则引擎] 组件{comp_id}在 '{room_name}' 区域外")
            
            # 如果找到多个组件，保留最大的，移除较小的
            if len(nested_components) > 1:
                print(f"   ⚠️ [空间规则引擎] 发现 '{room_name}' 内有 {len(nested_components)} 个卧室组件，需要清理嵌套")
                
                # 找到最大的组件
                largest_comp = max(nested_components, key=lambda comp_id: stats[comp_id, cv2.CC_STAT_AREA])
                largest_area = stats[largest_comp, cv2.CC_STAT_AREA]
                
                print(f"   📏 [空间规则引擎] 保留最大组件{largest_comp} (面积:{largest_area})")
                
                # 移除其他较小的组件
                for comp_id in nested_components:
                    if comp_id != largest_comp:
                        area = stats[comp_id, cv2.CC_STAT_AREA]
                        results[labels_im == comp_id] = 0  # 清除该组件
                        print(f"   🗑️ [空间规则引擎] 移除 '{room_name}' 内嵌套卧室组件{comp_id} (面积:{area})")
            elif len(nested_components) == 1:
                print(f"   ✅ [空间规则引擎] '{room_name}' 内只有1个卧室组件，无需清理")
            else:
                print(f"   ⚠️ [空间规则引擎] '{room_name}' 内没有AI分割的卧室组件")
        
        return results
    
    def _check_room_overlap(self, results, ocr_results):
        """检查房间重叠冲突，优先保留有OCR支持的房间"""
        print("   🔍 [空间规则引擎] 检查房间重叠冲突...")
        
        # 获取所有OCR支持的房间信息
        ocr_rooms = {}
        room_type_map = {
            "厨房": 7, "kitchen": 7,
            "卫生间": 2, "bathroom": 2, "washroom": 2,
            "客厅": 3, "living": 3,
            "卧室": 4, "bedroom": 4,
            "阳台": 6, "balcony": 6,
            "书房": 8, "study": 8
        }
        
        for item in ocr_results:
            text = item["text"].lower().strip()
            room_type = None
            
            for keyword, label in room_type_map.items():
                if keyword in text:
                    room_type = label
                    break
            
            if room_type:
                x, y, w, h = item["bbox"]
                # 转换到512x512坐标系
                ocr_to_512_scale_x = 512.0 / (item.get('ocr_width', 1158))
                ocr_to_512_scale_y = 512.0 / (item.get('ocr_height', 866))
                
                center_x_512 = int((x + w//2) * ocr_to_512_scale_x)
                center_y_512 = int((y + h//2) * ocr_to_512_scale_y)
                
                if room_type not in ocr_rooms:
                    ocr_rooms[room_type] = []
                ocr_rooms[room_type].append({
                    'center': (center_x_512, center_y_512),
                    'text': text,
                    'confidence': item.get('confidence', 1.0)
                })
        
        # 检查无OCR支持的大面积区域
        room_labels = [2, 3, 4, 6, 7, 8]  # 所有房间类型
        for label in room_labels:
            mask = (results == label)
            if not np.any(mask):
                continue
                
            # 如果有OCR支持，跳过检查
            if label in ocr_rooms and len(ocr_rooms[label]) > 0:
                continue
                
            # 计算无OCR支持区域的面积
            area = np.sum(mask)
            total_area = results.shape[0] * results.shape[1]
            area_ratio = area / total_area
            
            # 如果无OCR支持的区域过大，移除它
            if area_ratio > 0.08:  # 超过8%的无OCR支持区域（从15%调整）
                room_name = {2: "卫生间", 3: "客厅", 4: "卧室", 6: "阳台", 7: "厨房", 8: "书房"}[label]
                print(f"   🗑️ [空间规则引擎] 移除过大的无OCR支持{room_name}区域: {area_ratio:.1%}")
                results[mask] = 0  # 清除该区域
        
        # 检查房间重叠冲突
        results = self._check_room_overlap_conflicts(results)
        
        return results
    
    def _check_kitchen_position(self, results, ocr_results):
        """检查厨房位置合理性"""
        print("   🍳 [空间规则引擎] 检查厨房位置合理性...")
        
        # 获取客厅和厨房的OCR位置
        living_room_centers = []
        kitchen_centers = []
        
        for item in ocr_results:
            text = item["text"].lower().strip()
            x, y, w, h = item["bbox"]
            center_x, center_y = x + w//2, y + h//2
            
            if any(keyword in text for keyword in ["客厅", "living"]):
                living_room_centers.append((center_x, center_y))
            elif any(keyword in text for keyword in ["厨房", "kitchen"]):
                kitchen_centers.append((center_x, center_y))
        
        # 如果有客厅，检查厨房是否在客厅中央
        if living_room_centers and kitchen_centers:
            for lr_x, lr_y in living_room_centers:
                for kit_x, kit_y in kitchen_centers:
                    distance = np.sqrt((lr_x - kit_x)**2 + (lr_y - kit_y)**2)
                    if distance < 50:  # 距离太近，可能是错误识别
                        print(f"   ⚠️ [空间规则引擎] 厨房距客厅过近 (距离:{distance:.1f}px)，需要验证")
        
        return results
    
    def _check_room_overlap_conflicts(self, results):
        """检查房间重叠冲突，移除不合理的大面积重叠区域"""
        print("   🔍 [空间规则引擎] 检查房间重叠冲突...")
        
        room_labels = [2, 3, 4, 6, 7, 8]  # 卫生间、客厅、卧室、阳台、厨房、书房
        room_names = {2: "卫生间", 3: "客厅", 4: "卧室", 6: "阳台", 7: "厨房", 8: "书房"}
        
        # 检查每种房间类型的连通域
        for label in room_labels:
            mask = (results == label).astype(np.uint8)
            if not np.any(mask):
                continue
                
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            
            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                total_area = results.shape[0] * results.shape[1]
                area_ratio = area / total_area
                
                # 检查大面积房间与其他房间的重叠
                if area_ratio > 0.15:  # 面积超过15%的房间需要重叠检查
                    component_mask = (labels_im == comp_id)
                    overlap_count = 0
                    overlap_types = []
                    
                    # 检查与其他房间类型的重叠
                    for other_label in room_labels:
                        if other_label == label:
                            continue
                            
                        other_mask = (results == other_label)
                        if not np.any(other_mask):
                            continue
                            
                        # 计算重叠区域
                        overlap_area = np.sum(component_mask & other_mask)
                        overlap_ratio = overlap_area / area if area > 0 else 0
                        
                        if overlap_ratio > 0.1:  # 重叠超过10%
                            overlap_count += 1
                            overlap_types.append(room_names[other_label])
                    
                    # 如果与多个房间重叠，移除该区域
                    if overlap_count >= 2:
                        print(f"   🗑️ [空间规则引擎] 移除多重叠{room_names[label]}区域 (面积:{area_ratio:.1%}, 重叠:{overlap_count}个房间: {', '.join(overlap_types)})")
                        results[component_mask] = 0
                    elif overlap_count == 1 and area_ratio > 0.25:  # 单个重叠但面积过大
                        print(f"   🗑️ [空间规则引擎] 移除过大的重叠{room_names[label]}区域 (面积:{area_ratio:.1%}, 与{overlap_types[0]}重叠)")
                        results[component_mask] = 0
        
        return results


class SizeConstraintEngine:
    """尺寸约束引擎"""
    
    def validate_size_constraints(self, results, original_size):
        """验证尺寸约束"""
        print("📏 [尺寸约束引擎] 验证房间尺寸...")
        
        # 计算像素到实际尺寸的转换比例（基于常见户型图）
        # 假设图像宽度对应实际10-15米
        pixel_to_meter = 12.0 / original_size[0]  # 粗略估算
        
        # 首先检查大面积区域的合理性
        results = self._validate_large_area_rooms(results)
        
        # 检查各房间类型的尺寸合理性
        room_names = {2: "卫生间", 3: "客厅", 4: "卧室", 6: "阳台", 7: "厨房", 8: "书房"}
        
        for room_label, room_name in room_names.items():
            if room_label in [2, 7]:  # 重点检查卫生间和厨房
                results = self._check_room_size(results, room_label, room_name, pixel_to_meter)
        
        return results
    
    def _check_room_size(self, results, room_label, room_name, pixel_to_meter):
        """检查特定房间类型的尺寸"""
        print(f"   📐 [尺寸约束引擎] 检查{room_name}尺寸...")
        
        mask = (results == room_label).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        # 设定合理的面积范围（平方米）
        if room_label == 2:  # 卫生间
            min_area_m2, max_area_m2 = 2, 15  # 2-15平方米
        elif room_label == 7:  # 厨房
            min_area_m2, max_area_m2 = 3, 20  # 3-20平方米
        else:
            return results
        
        for comp_id in range(1, num_labels):
            area_pixels = stats[comp_id, cv2.CC_STAT_AREA]
            area_m2 = area_pixels * (pixel_to_meter ** 2)
            
            if area_m2 > max_area_m2:
                print(f"   ⚠️ [尺寸约束引擎] {room_name}过大: {area_m2:.1f}m² (>{max_area_m2}m²), 需要修正")
                # 移除过大的区域
                results[labels_im == comp_id] = 0
                print(f"   🗑️ [尺寸约束引擎] 移除过大{room_name}区域")
            elif area_m2 < min_area_m2:
                print(f"   ⚠️ [尺寸约束引擎] {room_name}过小: {area_m2:.1f}m² (<{min_area_m2}m²), 可能是误识别")
        
        return results
    
    def _validate_large_area_rooms(self, results):
        """验证大面积房间的合理性"""
        print("   📐 [尺寸约束引擎] 检查大面积区域合理性...")
        
        total_area = results.shape[0] * results.shape[1]
        room_names = {2: "卫生间", 3: "客厅", 4: "卧室", 6: "阳台", 7: "厨房", 8: "书房"}
        
        # 不同房间类型的合理面积上限（面积比例）
        max_ratios = {
            2: 0.10,    # 卫生间最多10%
            3: 0.40,    # 客厅最多40%
            4: 0.30,    # 单个卧室最多30%
            6: 0.15,    # 阳台最多15%
            7: 0.15,    # 厨房最多15%
            8: 0.30     # 书房最多5%（严格限制）
        }
        
        for label, room_name in room_names.items():
            mask = (results == label).astype(np.uint8)
            if not np.any(mask):
                continue
                
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            max_ratio = max_ratios.get(label, 0.25)
            
            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                area_ratio = area / total_area
                
                if area_ratio > max_ratio:
                    print(f"   🗑️ [尺寸约束引擎] 移除过大{room_name}区域: {area_ratio:.1%} > {max_ratio:.1%}")
                    results[labels_im == comp_id] = 0
        
        return results


class BuildingBoundaryDetector:
    """建筑边界检测器"""
    
    def validate_building_boundary(self, results, original_size):
        """验证建筑边界"""
        print("🏗️ [边界检测器] 验证建筑边界...")
        
        # 检测图像边缘的房间标记（可能是外部标尺误识别）
        results = self._remove_edge_misidentifications(results)
        
        return results
    
    def _remove_edge_misidentifications(self, results):
        """移除边缘位置的误识别"""
        print("   🚫 [边界检测器] 检查边缘误识别...")
        
        h, w = results.shape
        edge_threshold = 20  # 边缘阈值像素
        
        # 检查四个边缘区域
        edges = [
            (0, edge_threshold, 0, w),  # 上边缘
            (h-edge_threshold, h, 0, w),  # 下边缘  
            (0, h, 0, edge_threshold),  # 左边缘
            (0, h, w-edge_threshold, w)  # 右边缘
        ]
        
        for y1, y2, x1, x2 in edges:
            edge_region = results[y1:y2, x1:x2]
            unique_labels = np.unique(edge_region)
            
            # 移除边缘区域的房间标记（除了背景0和墙体1）
            for label in unique_labels:
                if label > 1:  # 房间标签
                    room_pixels = np.sum(edge_region == label)
                    if room_pixels > 50:  # 如果边缘区域有较多该房间像素
                        print(f"   🗑️ [边界检测器] 移除边缘区域房间标记 (标签:{label}, 像素:{room_pixels})")
                        results[results == label] = 0
        
        return results


class FloorplanProcessor:
    """户型图处理器 - 四层智能决策架构统一管理器"""

    def __init__(self, model_path="pretrained"):
        """初始化四层架构处理器"""
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

                    # 标注房间中心点
                    ax.plot(center_x, center_y, "o", markersize=10, color="white",
                           markeredgecolor="black", markeredgewidth=2)

                    # 房间标注
                    if len(room_list) > 1:
                        label_text = f"{room_type}{i+1}\n({center_x},{center_y})"
                    else:
                        label_text = f"{room_type}\n({center_x},{center_y})"

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
        print(f"✅ 边界重构完成: 新增墙体 {added} 像素, 清理噪点 {removed} 像素, 大块组件 {num-1} -> {np.unique(labels2).size-1}")
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
                keyword in text for keyword in ["卧室", "bedroom", "主卧", "次卧"]
            ):
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
                # 使用OCR文字的中心位置
                x, y, w, h = item["bbox"]

                # 计算OCR文字中心（在OCR处理的图像坐标系中）
                ocr_center_x = x + w // 2
                ocr_center_y = y + h // 2

                # OCR图像是放大2倍的，需要先转换到原始图像坐标
                orig_center_x = int(ocr_center_x / 2)
                orig_center_y = int(ocr_center_y / 2)

                # 优先使用分割掩码确定整间房的边界
                min_x = max_x = min_y = max_y = None
                label = room_label_mapping.get(room_type)
                if label is not None:
                    mask = enhanced_resized == label
                    mask_h, mask_w = mask.shape
                    mask_x = int(orig_center_x * mask_w / original_width)
                    mask_y = int(orig_center_y * mask_h / original_height)
                    seed_x, seed_y, seed_found = mask_x, mask_y, False

                    if 0 <= mask_x < mask_w and 0 <= mask_y < mask_h:
                        if mask[mask_y, mask_x]:
                            seed_found = True
                        else:
                            # 在附近寻找最近的同标签像素（7x7邻域）
                            search_radius = 3
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
                        labeled_mask = mask.astype(np.uint8)
                        num_labels, labels_img = cv2.connectedComponents(labeled_mask)
                        region_label = labels_img[seed_y, seed_x]
                        if region_label != 0:
                            region = labels_img == region_label
                            y_coords, x_coords = np.where(region)
                            min_x_512, max_x_512 = x_coords.min(), x_coords.max()
                            min_y_512, max_y_512 = y_coords.min(), y_coords.max()
                            scale_x = original_width / float(mask_w)
                            scale_y = original_height / float(mask_h)
                            min_x = int(min_x_512 * scale_x)
                            max_x = int(max_x_512 * scale_x)
                            min_y = int(min_y_512 * scale_y)
                            max_y = int(max_y_512 * scale_y)

                if min_x is None:
                    # 未找到连通域，回退到基于OCR文字的最小边界
                    orig_w = int(w / 2)  # OCR宽度转换到原始图像
                    orig_h = int(h / 2)  # OCR高度转换到原始图像
                    half_width = max(20, orig_w // 2)
                    half_height = max(15, orig_h // 2)
                    min_x = max(0, orig_center_x - half_width)
                    max_x = min(original_width - 1, orig_center_x + half_width)
                    min_y = max(0, orig_center_y - half_height)
                    max_y = min(original_height - 1, orig_center_y + half_height)

                width = max_x - min_x + 1
                height = max_y - min_y + 1

                room_info[room_type].append({
                    'center': (orig_center_x, orig_center_y),
                    'bbox': (min_x, min_y, max_x, max_y),
                    'pixels': width * height,  # 基于边界框的面积
                    'width': width,
                    'height': height,
                    'text': text,
                    'confidence': item.get('confidence', 0.0),
                })
        # 对于没有OCR检测到的房间，尝试从分割结果中提取
        label_mapping = {v: k for k, v in room_label_mapping.items()}

        for label, room_type in label_mapping.items():
            if len(room_info[room_type]) == 0:  # OCR没有检测到
                mask = enhanced_resized == label
                pixels = np.sum(mask)

                if pixels > 0:
                    # 计算面积比例，防止无OCR支持的过大区域
                    total_pixels = enhanced_resized.shape[0] * enhanced_resized.shape[1]
                    area_ratio = pixels / total_pixels
                    
                    # 对于没有OCR支持的房间，限制最大面积
                    max_area_without_ocr = 0.05  # 最多5% (从10%调整)
                    if area_ratio > max_area_without_ocr:
                        print(f"⚠️ [第3层-融合决策器] 跳过过大的无OCR支持{room_type}区域: {area_ratio:.1%} > {max_area_without_ocr:.1%}")
                        continue
                    # 找到房间区域的坐标
                    coords = np.where(mask)
                    y_coords, x_coords = coords

                    # 计算边界框
                    min_x_512, max_x_512 = np.min(x_coords), np.max(x_coords)
                    min_y_512, max_y_512 = np.min(y_coords), np.max(y_coords)

                    # 计算中心点
                    center_x_512 = int(np.mean(x_coords))
                    center_y_512 = int(np.mean(y_coords))

                    # 转换到原始图像尺寸
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

                    room_info[room_type].append(
                        {
                            "center": (center_x, center_y),
                            "bbox": (min_x, min_y, max_x, max_y),
                            "pixels": pixels,
                            "width": width,
                            "height": height,
                            "text": "分割检测",
                            "confidence": 0.5,
                        }
                    )

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
        
        return room_info
        
    def _merge_nearby_rooms(self, room_info, original_size):
        """合并距离很近的同类型房间"""
        print("🔄 检查并合并相近的同类型房间...")
        
        # 定义合并距离阈值（像素）
        merge_threshold = 50  # 中心点距离小于50像素的认为是同一房间
        
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
                    
                    if distance < merge_threshold:
                        to_merge.append(room2)
                        processed.add(j)
                        print(f"   🔗 {room_type}合并：'{room1['text']}'({x1},{y1}) + '{room2['text']}'({x2},{y2}) 距离{distance:.1f}像素")
                
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
