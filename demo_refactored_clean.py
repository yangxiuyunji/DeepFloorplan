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
import argparse
import numpy as np
from pathlib import Path

# 配置环境
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

# 导入原有工具模块
from utils.ocr_enhanced import extract_room_text, fuse_ocr_and_segmentation
from utils.rgb_ind_convertor import floorplan_fuse_map_figure
from room_detection_manager import RefactoredRoomDetectionManager


class FloorplanProcessor:
    """户型图处理器 - 统一管理整个处理流程"""
    
    def __init__(self, model_path="pretrained"):
        """初始化处理器"""
        self.model_path = model_path
        self.session = None
        self.inputs = None
        self.room_type_logit = None
        self.room_boundary_logit = None
        self.room_manager = RefactoredRoomDetectionManager()
        self.last_enhanced = None  # 用于摘要统计
        
        print("🏠 DeepFloorplan 房间检测 - 重构版本 (带坐标轴)")
        print("="*60)
        
    def load_model(self):
        """加载神经网络模型"""
        print("🔧 加载DeepFloorplan模型...")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        
        self.session = tf.Session(config=config)
        
        # 加载模型
        saver = tf.train.import_meta_graph(f"{self.model_path}/pretrained_r3d.meta")
        saver.restore(self.session, f"{self.model_path}/pretrained_r3d")
        
        # 获取输入输出节点
        graph = tf.get_default_graph()
        self.inputs = graph.get_tensor_by_name("inputs:0")
        self.room_type_logit = graph.get_tensor_by_name("Cast:0")
        self.room_boundary_logit = graph.get_tensor_by_name("Cast_1:0")
        
        print("✅ 模型加载完成")
        
    def preprocess_image(self, image_path):
        """图像预处理"""
        print(f"📸 处理图像: {image_path}")
        
        # 读取图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        print(f"📏 原始图像尺寸: {original_size[0]} x {original_size[1]} (宽x高)")
        
        # 调整到模型输入尺寸 (512x512)
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        
        print(f"🔄 神经网络输入: 512 x 512 (固定尺寸)")
        
        return img_array, original_size, np.array(img)
        
    def run_inference(self, img_array):
        """运行神经网络推理"""
        print("🤖 运行神经网络推理...")
        
        # 扩展维度以适应批处理
        input_batch = np.expand_dims(img_array, axis=0)
        
        # 运行推理
        room_type_logit, room_boundary_logit = self.session.run(
            [self.room_type_logit, self.room_boundary_logit], 
            feed_dict={self.inputs: input_batch}
        )
        
        # 合并房间类型和边界预测
        logits = np.concatenate([room_type_logit, room_boundary_logit], axis=-1)
        
        # 获取分割结果
        prediction = np.squeeze(np.argmax(logits, axis=-1))
        
        print("✅ 神经网络推理完成")
        return prediction
        
    def extract_ocr_info(self, original_img):
        """提取OCR文字信息"""
        print("🔍 提取OCR文字信息...")
        
        # OCR处理（放大2倍提高识别率）
        ocr_img = cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        print(f"🔍 OCR处理图像: {ocr_img.shape[1]} x {ocr_img.shape[0]} (放大2倍)")
        
        # 提取房间文字
        room_text_items = extract_room_text(ocr_img)
        
        print(f"📊 PaddleOCR检测到 {len(room_text_items)} 个文字区域")
        
        return room_text_items, ocr_img.shape
        
    def fuse_predictions(self, prediction, room_text_items, ocr_shape):
        """融合神经网络预测和OCR结果"""
        print("🔗 融合神经网络预测和OCR结果...")
        
        # 计算坐标转换比例
        ocr_to_512_x = 512.0 / ocr_shape[1]  
        ocr_to_512_y = 512.0 / ocr_shape[0]
        
        print(f"   🔄 OCR坐标转换到512x512:")
        print(f"      OCR图像({ocr_shape[1]}x{ocr_shape[0]}) -> 512x512")
        print(f"      转换比例: X={ocr_to_512_x:.3f}, Y={ocr_to_512_y:.3f}")
        
        # 转换OCR坐标到512x512坐标系
        converted_items = []
        for item in room_text_items:
            # 复制item并转换坐标
            converted_item = item.copy()
            x, y, w, h = item['bbox']
            
            # 转换到512x512坐标系
            new_x = int(x * ocr_to_512_x)
            new_y = int(y * ocr_to_512_y)
            new_w = int(w * ocr_to_512_x)
            new_h = int(h * ocr_to_512_y)
            
            # 确保坐标在512x512范围内
            new_x = max(0, min(new_x, 511))
            new_y = max(0, min(new_y, 511))
            new_w = max(1, min(new_w, 512 - new_x))
            new_h = max(1, min(new_h, 512 - new_y))
            
            converted_item['bbox'] = [new_x, new_y, new_w, new_h]
            converted_items.append(converted_item)
        
        # 融合OCR标签到分割结果
        enhanced = fuse_ocr_and_segmentation(prediction.copy(), converted_items)
        
        return enhanced
        
    def detect_rooms(self, enhanced, room_text_items, original_size):
        """检测各类房间"""
        print("🏠 开始房间检测...")
        
        # 使用重构管理器的统一接口
        enhanced = self.room_manager.detect_all_rooms(enhanced, room_text_items)
        
        return enhanced
        
    def generate_results(self, enhanced, original_img, original_size, output_path):
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
            mask = (enhanced_resized == label_value)
            colored_result[mask] = color
        
        # 叠加到原图
        alpha = 0.5
        final_result = cv2.addWeighted(original_img, 1-alpha, colored_result, alpha, 0)
        
        # 使用matplotlib创建带坐标轴的图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：原图 + 分割结果
        ax1.imshow(final_result)
        ax1.set_title('房间检测结果', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X坐标 (像素)', fontsize=12)
        ax1.set_ylabel('Y坐标 (像素)', fontsize=12)
        
        # 添加房间标注和坐标
        room_info = self._extract_room_coordinates(enhanced_resized, original_size, self.last_room_text_items)
        for room_type, room_list in room_info.items():
            for i, coords in enumerate(room_list):
                if coords['pixels'] > 0:  # 只显示有效检测的房间
                    center_x, center_y = coords['center']
                    bbox = coords['bbox']
                    
                    # 在图上标注房间中心点
                    ax1.plot(center_x, center_y, 'o', markersize=10, 
                            color='white', markeredgecolor='black', markeredgewidth=2)
                    
                    # 房间标注（如果有多个同类型房间，加上编号）
                    if len(room_list) > 1:
                        label_text = f'{room_type}{i+1}\n({center_x},{center_y})'
                    else:
                        label_text = f'{room_type}\n({center_x},{center_y})'
                        
                    ax1.annotate(label_text, 
                               xy=(center_x, center_y), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, edgecolor='red', linewidth=2, linestyle='--')
                    ax1.add_patch(rect)
        
        # 右图：纯分割结果
        ax2.imshow(colored_result)
        ax2.set_title('分割标签图', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X坐标 (像素)', fontsize=12)
        ax2.set_ylabel('Y坐标 (像素)', fontsize=12)
        
        # 添加图例
        legend_elements = []
        room_colors = {
            7: ('厨房', 'green'),
            2: ('卫生间', 'blue'),
            3: ('客厅', 'orange'),
            4: ('卧室', 'purple'),
            6: ('阳台', 'cyan'),
            8: ('书房', 'brown'),
            9: ('墙体', 'gray'),
            10: ('墙体', 'gray')
        }
        
        for label, (name, color) in room_colors.items():
            if np.any(enhanced_resized == label):
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                                markerfacecolor=color, markersize=10, label=f'{name} (标签{label})'))
        
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存结果
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        base_name = Path(output_path).stem
        
        # 保存带坐标轴的图像
        coordinate_result_path = output_dir / f"{base_name}_coordinate_result.png"
        plt.savefig(coordinate_result_path, dpi=300, bbox_inches='tight')
        print(f"📊 带坐标轴结果已保存: {coordinate_result_path}")
        
        # 保存原始结果（保持兼容性）
        result_path = output_dir / f"{base_name}_result.png"
        cv2.imwrite(str(result_path), cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
        print(f"📸 标准结果已保存: {result_path}")
        
        # 输出房间坐标信息
        self._print_room_coordinates(room_info, original_size)
        
        plt.close()
        
        return final_result
    
    def _extract_room_coordinates(self, enhanced_resized, original_size, room_text_items):
        """提取各房间的坐标信息，优先使用OCR文字位置，支持多个同类型房间"""
        room_info = {}
        
        # 计算坐标转换比例
        original_width, original_height = original_size
        
        # 定义房间类型
        room_types = ['厨房', '卫生间', '客厅', '卧室', '阳台', '书房']
        
        # 初始化所有房间信息为空列表，支持多个同类型房间
        for room_type in room_types:
            room_info[room_type] = []
        
        # 优先使用OCR文字位置确定房间坐标
        for item in room_text_items:
            text = item['text'].lower().strip()
            
            # 匹配房间类型
            room_type = None
            if any(keyword in text for keyword in ['厨房', 'kitchen', '厨']):
                room_type = '厨房'
            elif any(keyword in text for keyword in ['卫生间', 'bathroom', '卫', '洗手间', '浴室', '淋浴间', 'shower', '淋浴', '盥洗室']):
                room_type = '卫生间'  
            elif any(keyword in text for keyword in ['客厅', 'living', '厅', '起居室']):
                room_type = '客厅'
            elif any(keyword in text for keyword in ['卧室', 'bedroom', '主卧', '次卧']):
                room_type = '卧室'
            elif any(keyword in text for keyword in ['阳台', 'balcony']):
                room_type = '阳台'
            elif any(keyword in text for keyword in ['书房', 'study', '书', '办公室', 'office']):
                room_type = '书房'
            
            if room_type and room_type in room_info:
                # 使用OCR文字的中心位置
                x, y, w, h = item['bbox']
                
                # 计算OCR文字中心（在OCR处理的图像坐标系中）
                ocr_center_x = x + w // 2
                ocr_center_y = y + h // 2
                
                # OCR图像是放大2倍的，需要先转换到原始图像坐标
                orig_center_x = int(ocr_center_x / 2)
                orig_center_y = int(ocr_center_y / 2)
                
                # 计算边界框（基于文字位置估算房间区域）
                text_width = max(50, w // 2)  # 最小50像素宽度
                text_height = max(30, h // 2)  # 最小30像素高度
                
                min_x = max(0, orig_center_x - text_width)
                max_x = min(original_width - 1, orig_center_x + text_width)
                min_y = max(0, orig_center_y - text_height)
                max_y = min(original_height - 1, orig_center_y + text_height)
                
                room_info[room_type].append({
                    'center': (orig_center_x, orig_center_y),
                    'bbox': (min_x, min_y, max_x, max_y),
                    'pixels': text_width * text_height * 2,  # 估算面积
                    'width': max_x - min_x + 1,
                    'height': max_y - min_y + 1,
                    'text': text,
                    'confidence': item.get('confidence', 0.0)
                })
        
        # 对于没有OCR检测到的房间，尝试从分割结果中提取
        label_mapping = {7: '厨房', 2: '卫生间', 3: '客厅', 4: '卧室', 6: '阳台', 8: '书房'}
        
        for label, room_type in label_mapping.items():
            if len(room_info[room_type]) == 0:  # OCR没有检测到
                mask = (enhanced_resized == label)
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
                    
                    room_info[room_type].append({
                        'center': (center_x, center_y),
                        'bbox': (min_x, min_y, max_x, max_y),
                        'pixels': pixels,
                        'width': max_x - min_x + 1,
                        'height': max_y - min_y + 1,
                        'text': '分割检测',
                        'confidence': 0.5
                    })
        
        return room_info
    
    def _print_room_coordinates(self, room_info, original_size):
        """打印房间坐标详细信息，支持多个同类型房间"""
        print("\n" + "="*60)
        print("📍 房间坐标详细信息")
        print("="*60)
        print(f"📏 图像尺寸: {original_size[0]} x {original_size[1]} (宽 x 高)")
        print("-"*60)
        
        total_rooms = 0
        for room_type, room_list in room_info.items():
            if len(room_list) > 0:
                for i, info in enumerate(room_list):
                    if info['pixels'] > 0:
                        center_x, center_y = info['center']
                        min_x, min_y, max_x, max_y = info['bbox']
                        
                        # 如果有多个同类型房间，显示编号
                        if len(room_list) > 1:
                            display_name = f"{room_type}{i+1}"
                        else:
                            display_name = room_type
                            
                        print(f"🏠 {display_name}:")
                        print(f"   📍 中心坐标: ({center_x}, {center_y})")
                        print(f"   📐 边界框: 左上({min_x}, {min_y}) -> 右下({max_x}, {max_y})")
                        print(f"   📏 尺寸: {info['width']} x {info['height']} 像素")
                        print(f"   📊 面积: {info['pixels']} 像素")
                        print(f"   📄 识别文本: '{info['text']}' (置信度: {info['confidence']:.3f})")
                        print(f"   🔗 坐标范围: X[{min_x}-{max_x}], Y[{min_y}-{max_y}]")
                        print("-"*60)
                        total_rooms += 1
            
            # 如果该类型房间未检测到
            if len(room_list) == 0:
                print(f"❌ {room_type}: 未检测到")
                print("-"*60)
        
        print("💡 坐标系说明:")
        print("   • 原点(0,0)在图像左上角")
        print("   • X轴向右为正方向") 
        print("   • Y轴向下为正方向")
        print("   • 所有坐标单位为像素")
        print("="*60)
        print(f"\n📊 总计检测到 {total_rooms} 个房间")
        print("="*60)
        
    def process(self, image_path, output_path=None):
        """完整处理流程"""
        try:
            # 设置输出路径
            if output_path is None:
                output_path = Path(image_path).stem
                
            # 1. 加载模型
            if self.session is None:
                self.load_model()
                
            # 2. 图像预处理
            img_array, original_size, original_img = self.preprocess_image(image_path)
            
            # 3. 神经网络推理
            prediction = self.run_inference(img_array)
            
            # 4. OCR文字提取
            room_text_items, ocr_shape = self.extract_ocr_info(original_img)
            
            # 保存room_text_items用于坐标提取
            self.last_room_text_items = room_text_items
            
            # 5. 融合预测结果
            enhanced = self.fuse_predictions(prediction, room_text_items, ocr_shape)
            
            # 6. 房间检测
            enhanced = self.detect_rooms(enhanced, room_text_items, original_size)
            
            # 7. 生成结果
            result = self.generate_results(enhanced, original_img, original_size, output_path)
            
            # 8. 保存结果用于摘要统计
            self.last_enhanced = enhanced
            
            # 9. 显示摘要
            self._print_summary()
            
            return result
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            raise
            
    def _print_summary(self):
        """打印检测摘要"""
        # 统计检测到的房间数量
        kitchen_count = 1 if np.any(self.last_enhanced == 7) else 0
        bathroom_count = 1 if np.any(self.last_enhanced == 2) else 0  
        living_count = 1 if np.any(self.last_enhanced == 3) else 0
        
        total_rooms = kitchen_count + bathroom_count + living_count
        
        print(f"\n🏠 检测摘要: {kitchen_count}个厨房 + "
              f"{bathroom_count}个卫生间 + "
              f"{living_count}个客厅 = {total_rooms}个房间")
        
        if kitchen_count > 0:
            print("🍳 厨房检测: 绿色标记")
        if bathroom_count > 0:
            print("🚿 卫生间检测: 蓝色标记")
        if living_count > 0:
            print("🏠 客厅检测: 橙色标记")
            
    def __del__(self):
        """清理资源"""
        if self.session:
            self.session.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DeepFloorplan 房间检测 - 重构版本 (带坐标轴)')
    parser.add_argument('image', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出文件名前缀')
    parser.add_argument('--model', '-m', default='pretrained', help='模型路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.image).exists():
        print(f"❌ 输入文件不存在: {args.image}")
        sys.exit(1)
        
    # 创建处理器并执行
    processor = FloorplanProcessor(args.model)
    result = processor.process(args.image, args.output)
    
    print("✅ 处理完成!")


if __name__ == "__main__":
    main()
