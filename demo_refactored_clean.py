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

# 配置环境
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image

# 配置中文字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
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
        
        room_type_logit, room_boundary_logit = self.session.run(
            [self.room_type_logit, self.room_boundary_logit],
            feed_dict={self.inputs: input_batch},
        )
        
        logits = np.concatenate([room_type_logit, room_boundary_logit], axis=-1)
        prediction = np.squeeze(np.argmax(logits, axis=-1))
        print("✅ [第1层-AI分割器] 神经网络推理完成")
        return prediction


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
        
        return room_text_items, ocr_img.shape


class FusionDecisionEngine:
    """第三层：融合决策器"""
    
    def __init__(self):
        self.room_manager = RefactoredRoomDetectionManager()
    
    def fuse_results(self, ai_prediction, ocr_results, ocr_shape):
        """智能融合AI分割和OCR识别结果"""
        print("🔗 [第3层-融合决策器] 融合AI和OCR结果...")
        
        # 坐标转换
        ocr_to_512_x = 512.0 / ocr_shape[1]
        ocr_to_512_y = 512.0 / ocr_shape[0]
        print(f"   🔄 [第3层-融合决策器] OCR坐标转换到512x512:")
        print(f"      OCR图像({ocr_shape[1]}x{ocr_shape[0]}) -> 512x512")
        print(f"      转换比例: X={ocr_to_512_x:.3f}, Y={ocr_to_512_y:.3f}")
        
        # 保存原始OCR结果（用于后续清理算法）
        original_ocr_results = [item.copy() for item in ocr_results]
        
        # 在OCR结果中添加尺寸信息，供后续使用
        for item in ocr_results:
            item['ocr_width'] = ocr_shape[1]
            item['ocr_height'] = ocr_shape[0]
        
        # 转换OCR坐标到512x512坐标系（用于神经网络融合）
        converted_items = self._convert_ocr_coordinates(ocr_results, ocr_to_512_x, ocr_to_512_y)

        # 拆分厨房OCR，用于开放式厨房估算
        processed_items = []
        open_kitchens = []
        for item in converted_items:
            label = text_to_label(item['text'])
            if label == 7:
                x, y, w, h = item['bbox']
                cx, cy = x + w // 2, y + h // 2
                if ai_prediction[cy, cx] == 3:
                    open_kitchens.append(item)
                    print(f"   🍳 识别到开放式厨房候选: {item['text']}")
                else:
                    processed_items.append(item)
            else:
                processed_items.append(item)

        # 融合OCR标签到分割结果（不含开放式厨房）
        enhanced = fuse_ocr_and_segmentation(ai_prediction.copy(), processed_items)

        # 开放式厨房区域估算
        enhanced = self._estimate_open_kitchen(enhanced, open_kitchens)
        
        # 房间检测和生成（使用原始OCR结果）
        enhanced = self.room_manager.detect_all_rooms(enhanced, original_ocr_results)
        
        # 添加OCR检测到的阳台区域标注
        enhanced = self._add_balcony_regions(enhanced, original_ocr_results, ocr_to_512_x, ocr_to_512_y)
        
        # 基础清理（使用原始OCR结果进行距离计算）
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
        """Estimate open kitchen areas when no wall is detected."""
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
            mask = ~np.isin(patch, [9, 10])
            patch[mask] = 7
            enhanced[y1:y2, x1:x2] = patch

        return enhanced
    
    def _add_balcony_regions(self, enhanced, original_ocr_results, scale_x, scale_y):
        """为OCR检测到的阳台添加分割标注"""
        print("🌞 [第3层-融合决策器] 添加阳台区域标注...")
        
        balcony_items = []
        for item in original_ocr_results:
            text = item["text"].lower().strip()
            if any(keyword in text for keyword in ["阳台", "balcony", "阳兮", "阳合", "阳囊"]):
                balcony_items.append(item)
                print(f"   🎯 发现阳台OCR: '{item['text']}' (置信度: {item['confidence']:.3f})")
        
        # 为每个检测到的阳台创建区域标注
        for item in balcony_items:
            x, y, w, h = item["bbox"]
            # 转换到512x512坐标系
            center_x_512 = int((x + w//2) * scale_x)
            center_y_512 = int((y + h//2) * scale_y)
            
            # 确保坐标在有效范围内
            center_x_512 = max(0, min(center_x_512, 511))
            center_y_512 = max(0, min(center_y_512, 511))
            
            # 创建阳台区域（使用适中的尺寸）
            balcony_size = 30  # 阳台通常比较小
            x1 = max(0, center_x_512 - balcony_size // 2)
            y1 = max(0, center_y_512 - balcony_size // 2)
            x2 = min(511, center_x_512 + balcony_size // 2)
            y2 = min(511, center_y_512 + balcony_size // 2)
            
            # 在该区域设置阳台标签（6）
            enhanced[y1:y2, x1:x2] = 6
            print(f"   ✅ 阳台区域标注: 中心({center_x_512}, {center_y_512}), 区域({x1}, {y1}) -> ({x2}, {y2})")
        
        return enhanced
    
    def _basic_cleanup(self, enhanced, original_ocr_results, scale_x, scale_y):
        """基础清理：距离阈值清理"""
        print("🧹 [第3层-融合决策器] 基础清理...")
        
        # 获取OCR验证的房间位置（使用原始坐标转换到512x512）
        ocr_rooms = self._extract_ocr_rooms_for_cleanup(original_ocr_results, scale_x, scale_y)
        
        # 清理误识别区域
        for room_label, room_positions in ocr_rooms.items():
            if room_label in [2, 3, 4, 7]:  # 处理卫生间、客厅、卧室和厨房
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
        """清理特定房间类型的误识别"""
        room_names = {2: "卫生间", 3: "客厅", 4: "卧室", 7: "厨房"}
        room_name = room_names.get(room_label, "房间")
        print(f"🧹 [第3层-融合决策器] 清理{room_name}误识别，保留{len(room_positions)}个OCR验证位置")
        
        mask = (enhanced == room_label).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        cleaned_mask = np.zeros_like(mask)
        
        for comp_id in range(1, num_labels):
            comp_centroid = centroids[comp_id]
            comp_center_x, comp_center_y = int(comp_centroid[0]), int(comp_centroid[1])
            comp_area = stats[comp_id, cv2.CC_STAT_AREA]
            
            # 计算到最近OCR位置的距离
            min_distance = float('inf')
            closest_confidence = 0
            for ocr_x, ocr_y, confidence in room_positions:
                distance = np.sqrt((comp_center_x - ocr_x)**2 + (comp_center_y - ocr_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_confidence = confidence
            
            # 根据房间类型设置动态阈值
            if room_label == 3:  # 客厅
                distance_threshold = 150  # 客厅允许更大的距离容错
                max_area_threshold = 25000  # 客厅面积上限更高
            elif room_label == 7:  # 厨房
                distance_threshold = 120 if len(room_positions) > 1 else 100
                max_area_threshold = 15000
            else:  # 卫生间、卧室等
                distance_threshold = 100 if len(room_positions) > 1 else 80
                max_area_threshold = 10000
            
            if min_distance < distance_threshold and comp_area < max_area_threshold:
                cleaned_mask[labels_im == comp_id] = 1
                print(f"   ✅ [第3层-融合决策器] 保留{room_name}区域：距OCR:{min_distance:.1f}px, 面积:{comp_area}")
            else:
                print(f"   ❌ [第3层-融合决策器] 移除{room_name}区域：距OCR:{min_distance:.1f}px, 面积:{comp_area}")
        
        enhanced[mask == 1] = 0
        enhanced[cleaned_mask == 1] = room_label
        
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
        
        # 规则2: 检查厨房位置合理性（不应在客厅中央）
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


class SizeConstraintEngine:
    """尺寸约束引擎"""
    
    def validate_size_constraints(self, results, original_size):
        """验证尺寸约束"""
        print("📏 [尺寸约束引擎] 验证房间尺寸...")
        
        # 计算像素到实际尺寸的转换比例（基于常见户型图）
        # 假设图像宽度对应实际10-15米
        pixel_to_meter = 12.0 / original_size[0]  # 粗略估算
        
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
        
        # 用于统计的变量
        self.last_enhanced = None

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
        img_array = np.array(img_resized) / 255.0

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
        return validated_results, ocr_results

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
            elif any(keyword in text for keyword in ["书房", "study", "书", "办公室", "office"]):
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

    def generate_results(
        self, enhanced, original_img, original_size, output_path, room_text_items
    ):
        """生成最终结果，包含坐标轴和房间坐标信息"""
        print("🎨 生成结果图像...")

        # 调整回原始尺寸
        enhanced_resized = cv2.resize(
            enhanced, original_size, interpolation=cv2.INTER_NEAREST
        )

        # 生成彩色分割图 - 使用颜色映射字典
        h, w = enhanced_resized.shape
        colored_result = np.zeros((h, w, 3), dtype=np.uint8)

        # 应用颜色映射
        for label_value, color in floorplan_fuse_map_figure.items():
            mask = enhanced_resized == label_value
            colored_result[mask] = color

        # 叠加到原图
        alpha = 0.5
        final_result = cv2.addWeighted(
            original_img, 1 - alpha, colored_result, alpha, 0
        )

        # 使用matplotlib创建带坐标轴的图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 左图：原图 + 分割结果
        ax1.imshow(final_result)
        ax1.set_title("房间检测结果", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("X坐标 (像素)", fontsize=12)
        ax1.set_ylabel("Y坐标 (像素)", fontsize=12)

        # 添加房间标注和坐标
        room_info = self._extract_room_coordinates(
            enhanced_resized, original_size, room_text_items
        )
        
        # 保存房间信息供摘要使用
        self.last_room_info = room_info
        
        for room_type, room_list in room_info.items():
            for i, coords in enumerate(room_list):
                if coords["pixels"] > 0:  # 只显示有效检测的房间
                    center_x, center_y = coords["center"]
                    bbox = coords["bbox"]

                    # 在图上标注房间中心点
                    ax1.plot(
                        center_x,
                        center_y,
                        "o",
                        markersize=10,
                        color="white",
                        markeredgecolor="black",
                        markeredgewidth=2,
                    )

                    # 房间标注（如果有多个同类型房间，加上编号）
                    if len(room_list) > 1:
                        label_text = f"{room_type}{i+1}\n({center_x},{center_y})"
                    else:
                        label_text = f"{room_type}\n({center_x},{center_y})"

                    ax1.annotate(
                        label_text,
                        xy=(center_x, center_y),
                        xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )

                    # 绘制边界框
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        linestyle="--",
                    )
                    ax1.add_patch(rect)

        # 右图：纯分割结果
        ax2.imshow(colored_result)
        ax2.set_title("分割标签图", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("X坐标 (像素)", fontsize=12)
        ax2.set_ylabel("Y坐标 (像素)", fontsize=12)

        for room_type, room_list in room_info.items():
            for i, coords in enumerate(room_list):
                if coords["pixels"] > 0:
                    center_x, center_y = coords["center"]
                    bbox = coords["bbox"]

                    ax2.plot(
                        center_x,
                        center_y,
                        "o",
                        markersize=10,
                        color="white",
                        markeredgecolor="black",
                        markeredgewidth=2,
                    )

                    if len(room_list) > 1:
                        label_text = f"{room_type}{i+1}\n({center_x},{center_y})"
                    else:
                        label_text = f"{room_type}\n({center_x},{center_y})"

                    ax2.annotate(
                        label_text,
                        xy=(center_x, center_y),
                        xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )

                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        linestyle="--",
                    )
                    ax2.add_patch(rect)

        # 添加图例 - 颜色与实际渲染一致
        legend_elements = []
        # 使用与floorplan_fuse_map_figure完全一致的颜色定义
        room_colors = {
            7: ("厨房", np.array([0, 255, 0]) / 255.0),      # 纯绿色 [0,255,0]
            2: ("卫生间", np.array([192, 255, 255]) / 255.0),  # 浅青色 [192,255,255]
            3: ("客厅", np.array([224, 255, 192]) / 255.0),   # 浅绿色 [224,255,192]
            4: ("卧室", np.array([255, 224, 128]) / 255.0),   # 浅黄色 [255,224,128]
            6: ("阳台", np.array([255, 224, 224]) / 255.0),   # 浅粉色 [255,224,224]
            8: ("书房", np.array([224, 224, 128]) / 255.0),   # 浅黄绿 [224,224,128]
            9: ("门窗", np.array([255, 60, 128]) / 255.0),    # 粉红色 [255,60,128]
            10: ("墙体", np.array([0, 0, 0]) / 255.0),        # 黑色 [0,0,0]
        }

        for label, (name, color) in room_colors.items():
            # 检查该房间类型是否在分割图中存在，或者在room_info中有检测记录
            room_detected_in_image = np.any(enhanced_resized == label)
            room_detected_by_ocr = False
            
            # 检查room_info中是否有对应房间类型的检测记录
            room_name_map = {7: "厨房", 2: "卫生间", 3: "客厅", 4: "卧室", 6: "阳台", 8: "书房"}
            if label in room_name_map and room_name_map[label] in room_info:
                room_detected_by_ocr = len(room_info[room_name_map[label]]) > 0
            
            if room_detected_in_image or room_detected_by_ocr:
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="s",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                        label=f"{name} (标签{label})",
                    )
                )

        ax2.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.3, 1))

        # 调整布局
        plt.tight_layout()

        # 保存结果
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        base_name = Path(output_path).stem

        # 保存带坐标轴的图像
        coordinate_result_path = output_dir / f"{base_name}_coordinate_result.png"
        plt.savefig(coordinate_result_path, dpi=300, bbox_inches="tight")
        print(f"📊 带坐标轴结果已保存: {coordinate_result_path}")

        # 保存原始结果（保持兼容性）
        result_path = output_dir / f"{base_name}_result.png"
        cv2.imwrite(str(result_path), cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
        print(f"📸 标准结果已保存: {result_path}")

        # 输出房间坐标信息
        self._print_room_coordinates(room_info, original_size)

        plt.close()

        return final_result

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
            elif any(
                keyword in text
                for keyword in ["书房", "study", "书", "办公室", "office"]
            ):
                room_type = "书房"

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
            enhanced, room_text_items = self.process_with_four_layer_architecture(
                img_array, original_img, original_size
            )

            # 7. 生成结果
            result = self.generate_results(
                enhanced, original_img, original_size, output_path, room_text_items
            )

            # 8. 显示摘要
            self._print_summary()

            return result

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

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.image).exists():
        print(f"❌ 输入文件不存在: {args.image}")
        sys.exit(1)

    # 创建处理器并执行
    processor = FloorplanProcessor(args.model)
    result = processor.process(args.image, args.output)
    
    # 确定输出文件路径
    output_base = args.output if args.output else Path(args.image).stem
    coordinate_result_path = f"output/{output_base}_coordinate_result.png"
    standard_result_path = f"output/{output_base}_result.png"
    
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
