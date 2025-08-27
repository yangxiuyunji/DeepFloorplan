import os
import argparse
import numpy as np

# 配置TensorFlow日志级别，完全静音冗长输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误
import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib
import cv2
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import opening, closing, square, disk

tf.logging.set_verbosity(tf.logging.ERROR)  # 减少TensorFlow日志

# Configure Chinese font support for matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Disable TF 2.x behavior for compatibility
tf.disable_v2_behavior()

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# OCR utilities
from utils.ocr_enhanced import extract_room_text, fuse_ocr_and_segmentation, set_closet_enabled
from utils.rgb_ind_convertor import floorplan_fuse_map, floorplan_fuse_map_figure

def imread(path, mode='RGB'):
    """Read image using PIL"""
    img = Image.open(path)
    if mode == 'RGB':
        img = img.convert('RGB')
    elif mode == 'L':
        img = img.convert('L')
    return np.array(img)

def imsave(path, img):
    """Save image using PIL"""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def imresize(img, size):
    """Resize image using PIL"""
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    if len(img.shape) == 3:
        h, w, c = size if len(size) == 3 else (*size, img.shape[2])
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((w, h))
        return np.array(img_resized)
    else:
        h, w = size
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((w, h))
        return np.array(img_resized)
from matplotlib import pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def draw_chinese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文文字
    """
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体
    try:
        # Windows系统字体路径
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

# Force CPU usage - disable GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('image_path', nargs='?', default='./demo/demo.jpg',
                    help='input image path')
parser.add_argument('--im_path', type=str, default=None,
                    help='input image path (alternative to positional argument)')
parser.add_argument('--enable_closet', action='store_true',
                    help='enable closet predictions (disabled by default)')
parser.add_argument('--disable_closet', action='store_true',
                    help='map closet predictions to background (deprecated, use --enable_closet instead)')


def simple_connected_components(mask):
    """Simple connected component labeling without scipy"""
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0
    
    def flood_fill(start_y, start_x, label):
        """Flood fill algorithm for connected components"""
        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if (y < 0 or y >= h or x < 0 or x >= w or 
                labels[y, x] != 0 or not mask[y, x]):
                continue
            
            labels[y, x] = label
            # Add 4-connected neighbors
            stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current_label += 1
                flood_fill(y, x, current_label)
    
    return labels, current_label

def apply_precise_kitchen_coordinates(floorplan, ocr_results, ori_shape):
    """精确分析OCR识别的厨房文字位置，详细输出坐标转换过程"""
    if not ocr_results:
        return floorplan, []
    
    # 查找厨房OCR结果
    kitchen_boxes = []
    for ocr_item in ocr_results:
        text = ocr_item['text'].lower()
        text_stripped = text.strip()
        
        # 厨房关键词匹配（与厨房检测函数保持一致）
        single_char_keywords = ['厨']
        multi_char_keywords = ['厨房', 'kitchen', 'cook', '烹饪', 'cooking']
        
        is_single_char_match = (text_stripped in single_char_keywords or 
                              any(text_stripped.startswith(char) for char in single_char_keywords))
        is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
        
        if is_single_char_match or is_multi_char_match:
            x, y, w, h = ocr_item['bbox']
            
            print(f"🔍 OCR厨房识别详细分析:")
            print(f"   检测到文字: '{ocr_item['text']}'")
            print(f"   置信度: {ocr_item['confidence']:.3f}")
            print(f"   🎯 OCR原始数据分析:")
            print(f"      边界框(x,y,w,h): ({x}, {y}, {w}, {h})")
            print(f"      文字区域: 左上角({x}, {y}) -> 右下角({x+w}, {y+h})")
            
            # 计算OCR检测到的"厨房"文字的精确中心
            ocr_center_x = x + w // 2
            ocr_center_y = y + h // 2
            
            print(f"   📍 OCR文字中心计算:")
            print(f"      中心X = {x} + {w}//2 = {ocr_center_x}")
            print(f"      中心Y = {y} + {h}//2 = {ocr_center_y}")
            print(f"      OCR文字中心: ({ocr_center_x}, {ocr_center_y}) [512x512坐标系]")
            
            # 转换为原始图像坐标
            orig_center_x = int(ocr_center_x * ori_shape[1] / 512)
            orig_center_y = int(ocr_center_y * ori_shape[0] / 512)
            
            print(f"   🔄 坐标系转换:")
            print(f"      原始图像尺寸: {ori_shape[1]} x {ori_shape[0]}")
            print(f"      512x512 -> 原始图像转换比例:")
            print(f"        X比例: 512 -> {ori_shape[1]} (×{ori_shape[1]/512:.3f})")
            print(f"        Y比例: 512 -> {ori_shape[0]} (×{ori_shape[0]/512:.3f})")
            print(f"      转换后原始图像坐标: ({orig_center_x}, {orig_center_y})")
            
            # 在OCR检测到的厨房文字位置标记厨房
            radius = 20
            kitchen_pixels = 0
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    new_y = ocr_center_y + dy
                    new_x = ocr_center_x + dx
                    if (0 <= new_y < 512 and 0 <= new_x < 512 and 
                        dx*dx + dy*dy <= radius*radius):
                        floorplan[new_y, new_x] = 7  # 厨房标签
                        kitchen_pixels += 1
            
            print(f"   ✅ 在OCR文字中心位置标记了{kitchen_pixels}个像素为厨房")
            print(f"   📊 厨房文字识别结果:")
            print(f"      512x512坐标: ({ocr_center_x}, {ocr_center_y})")
            print(f"      原始图像坐标: ({orig_center_x}, {orig_center_y})")
            print(f"      这就是OCR精确识别的'厨房'两字的中心位置")
            
            kitchen_boxes.append({
                'center': (ocr_center_x, ocr_center_y),
                'original_center': (orig_center_x, orig_center_y),
                'bbox': (x, y, w, h),
                'text': ocr_item['text'],
                'confidence': ocr_item['confidence']
            })
    
    return floorplan, kitchen_boxes


def expand_kitchen_region_from_center(floorplan, center_x, center_y, original_shape, target_size=None):
    """从厨房中心点向四周扩展，形成规则的矩形厨房区域"""
    print(f"🏠 智能厨房区域扩展: 中心({center_x}, {center_y})")
    
    h, w = original_shape[:2]
    
    # 如果没有指定目标大小，根据图像大小估算合理的厨房大小
    if target_size is None:
        # 厨房通常占总面积的8-15%
        total_area = h * w
        target_area = total_area * 0.12  # 12%的面积
        target_size = int(np.sqrt(target_area))
    
    # 确保厨房大小合理（不能太小也不能太大）
    min_size = min(h, w) // 8  # 最小尺寸
    max_size = min(h, w) // 3  # 最大尺寸
    target_size = max(min_size, min(target_size, max_size))
    
    print(f"   🎯 目标厨房尺寸: {target_size}x{target_size} 像素")
    
    # 创建厨房掩码
    kitchen_mask = np.zeros((h, w), dtype=bool)
    
    # 计算矩形边界（以中心点为中心的正方形）
    half_size = target_size // 2
    
    # 确保边界在图像范围内
    left = max(0, center_x - half_size)
    right = min(w, center_x + half_size)
    top = max(0, center_y - half_size)
    bottom = min(h, center_y + half_size)
    
    # 调整边界，尽量保持正方形
    width = right - left
    height = bottom - top
    
    if width < height:
        # 宽度不够，尝试扩展左右
        needed = height - width
        if left > needed // 2:
            left = max(0, left - needed // 2)
        if right < w - needed // 2:
            right = min(w, right + needed // 2)
    elif height < width:
        # 高度不够，尝试扩展上下
        needed = width - height
        if top > needed // 2:
            top = max(0, top - needed // 2)
        if bottom < h - needed // 2:
            bottom = min(h, bottom + needed // 2)
    
    # 标记厨房区域
    kitchen_mask[top:bottom, left:right] = True
    
    # 检查是否与墙壁冲突，如果是则收缩区域
    conflict_pixels = 0
    for y in range(top, bottom):
        for x in range(left, right):
            if floorplan[y, x] in [9, 10]:  # 墙壁
                kitchen_mask[y, x] = False
                conflict_pixels += 1
    
    # 如果冲突太多，收缩区域
    if conflict_pixels > (bottom - top) * (right - left) * 0.3:  # 超过30%冲突
        print(f"   ⚠️ 墙壁冲突过多({conflict_pixels}像素)，收缩厨房区域")
        # 收缩到更小的区域
        new_half = target_size // 3
        left = max(0, center_x - new_half)
        right = min(w, center_x + new_half)
        top = max(0, center_y - new_half)
        bottom = min(h, center_y + new_half)
        
        kitchen_mask.fill(False)
        kitchen_mask[top:bottom, left:right] = True
        
        # 再次检查墙壁冲突
        for y in range(top, bottom):
            for x in range(left, right):
                if floorplan[y, x] in [9, 10]:
                    kitchen_mask[y, x] = False
    
    expanded_pixels = np.sum(kitchen_mask)
    final_width = right - left
    final_height = bottom - top
    
    print(f"   ✅ 厨房区域生成完成:")
    print(f"      区域大小: {final_width}x{final_height} 像素")
    print(f"      有效面积: {expanded_pixels} 像素")
    print(f"      区域位置: ({left},{top}) 到 ({right},{bottom})")
    
    return kitchen_mask


def enhance_bathroom_detection(floorplan, ocr_results):
    """智能卫生间检测：优先使用OCR，确保精准识别卫生间，形成规则的矩形区域"""
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # 首先检查OCR是否检测到卫生间
    bathroom_ocr_items = []
    if ocr_results:
        for ocr_item in ocr_results:
            text = ocr_item['text'].lower()
            text_stripped = text.strip()
            
            # 定义关键词：单字符简写（完全匹配或开头匹配）+ 多字符关键词（包含匹配）
            single_char_keywords = ['卫', '洗', '浴']  # 中文简写
            multi_char_keywords = ['卫生间', 'bathroom', 'toilet', 'wc', '厕所', 
                                 '浴室', 'shower', 'bath', '洗手间', '卫浴', 
                                 'restroom', 'washroom']
            
            # 检查是否为单字符简写（完全匹配或以该字符开头，如"卫A"、"卫B"）
            is_single_char_match = (text_stripped in single_char_keywords or 
                                  any(text_stripped.startswith(char) for char in single_char_keywords))
            
            # 检查是否包含多字符关键词
            is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
            
            if is_single_char_match or is_multi_char_match:
                bathroom_ocr_items.append(ocr_item)
                print(f"🚿 OCR检测到卫生间文字: '{ocr_item['text']}' (置信度: {ocr_item['confidence']:.3f})")
    
    # 如果OCR检测到卫生间，优先使用OCR结果
    if bathroom_ocr_items:
        print("✅ 使用OCR检测的卫生间位置")
        
        # 记录已生成的卫生间位置，确保彼此有足够距离
        existing_bathrooms = []
        
        # 处理多个卫生间OCR结果（可能有主卫、客卫）
        for i, bathroom_ocr in enumerate(bathroom_ocr_items):
            x, y, w, h = bathroom_ocr['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            
            print(f"   📍 处理卫生间 {i+1}: '{bathroom_ocr['text']}' (置信度: {bathroom_ocr['confidence']:.3f})")
            print(f"   🎯 卫生间中心位置: ({center_x}, {center_y})")
            
            # 检查与已有卫生间的距离
            too_close = False
            min_distance = min(h, w) * 0.15  # 最小距离为图像尺寸的15%
            
            for existing_center in existing_bathrooms:
                distance = np.sqrt((center_x - existing_center[0])**2 + (center_y - existing_center[1])**2)
                if distance < min_distance:
                    print(f"   ⚠️ 卫生间{i+1}距离现有卫生间过近({distance:.1f}px < {min_distance:.1f}px)，跳过")
                    too_close = True
                    break
            
            if too_close:
                continue
                
            # 从OCR中心点生成规则的卫生间区域
            bathroom_mask = create_regular_bathroom_area(enhanced, center_x, center_y, h, w)
            enhanced[bathroom_mask] = 2  # 卫生间标签
            
            bathroom_pixels = np.sum(bathroom_mask)
            print(f"   ✅ 生成规则卫生间区域 {i+1}: {bathroom_pixels} 像素")
            
            # 记录此卫生间位置
            existing_bathrooms.append((center_x, center_y))
        
        return enhanced
    
    # 如果OCR没有检测到卫生间，使用空间分析
    print("📍 OCR未检测到卫生间，使用空间分析方法")
    
    # 查找可能的卫生间区域（通常较小且靠近墙体）
    potential_bathroom_mask = (enhanced == 3) | (enhanced == 1)  # 客厅或未分类区域
    
    if np.sum(potential_bathroom_mask) == 0:
        print("❌ 未发现潜在的卫生间区域")
        return enhanced
    
    try:
        # 连通组件分析
        labeled_regions, num_regions = simple_connected_components(potential_bathroom_mask)
        
        print(f"🔍 发现 {num_regions} 个潜在的卫生间区域")
        
        region_stats = []
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            area = np.sum(region_mask)
            
            # 卫生间通常面积较小
            total_area = h * w
            area_ratio = area / total_area
            
            # 计算区域的紧凑度（更接近方形的区域得分更高）
            y_coords, x_coords = np.where(region_mask)
            if len(y_coords) > 0:
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                compactness = area / bbox_area if bbox_area > 0 else 0
                
                # 计算区域中心
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                
                region_stats.append({
                    'id': region_id,
                    'area': area,
                    'area_ratio': area_ratio,
                    'compactness': compactness,
                    'center': (center_x, center_y),
                    'bbox': (min_x, min_y, max_x, max_y)
                })
        
        # 筛选符合卫生间特征的区域
        bathroom_candidates = []
        for stat in region_stats:
            # 卫生间特征：面积适中（0.5%-8%），紧凑度较高
            if (0.005 <= stat['area_ratio'] <= 0.08 and 
                stat['compactness'] >= 0.3):
                bathroom_candidates.append(stat)
                print(f"   🚿 发现卫生间候选区域: 面积比例={stat['area_ratio']:.3f}, 紧凑度={stat['compactness']:.3f}")
        
        # 对候选区域进行评分和选择
        if bathroom_candidates:
            # 按综合得分排序（优先紧凑度高、面积适中的区域）
            for candidate in bathroom_candidates:
                # 计算得分：紧凑度权重更高
                score = candidate['compactness'] * 0.7 + (1 - abs(candidate['area_ratio'] - 0.03)) * 0.3
                candidate['score'] = score
            
            bathroom_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 最多识别2个卫生间（主卫+客卫）
            selected_bathrooms = bathroom_candidates[:2]
            
            for i, bathroom in enumerate(selected_bathrooms):
                center_x, center_y = bathroom['center']
                print(f"   ✅ 选择卫生间 {i+1}: 中心({center_x}, {center_y}), 得分={bathroom['score']:.3f}")
                
                # 生成规则的卫生间区域
                region_mask = (labeled_regions == bathroom['id'])
                enhanced[region_mask] = 2  # 卫生间标签
                
                print(f"   🚿 标记卫生间区域 {i+1}: {bathroom['area']} 像素")
        else:
            print("❌ 未找到符合特征的卫生间区域")
    
    except Exception as e:
        print(f"❌ 卫生间空间分析失败: {str(e)}")
    
    return enhanced


def create_regular_bathroom_area(floorplan, center_x, center_y, img_h, img_w):
    """从中心点创建规则的矩形卫生间区域，严格限制在房间边界内"""
    h, w = floorplan.shape
    
    print(f"      🚿 智能生成卫生间区域: 中心({center_x}, {center_y})")
    
    # 首先检查中心点是否在有效区域（非墙壁）
    if floorplan[center_y, center_x] in [9, 10]:
        print(f"      ⚠️ 中心点在墙壁上，寻找附近的有效区域")
        # 寻找附近的非墙壁区域
        found_valid = False
        for radius in range(1, 8):  # 搜索范围较小（卫生间通常较小）
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    new_y, new_x = center_y + dy, center_x + dx
                    if (0 <= new_y < h and 0 <= new_x < w and 
                        floorplan[new_y, new_x] not in [9, 10]):
                        center_x, center_y = new_x, new_y
                        found_valid = True
                        break
                if found_valid:
                    break
            if found_valid:
                break
        
        if not found_valid:
            print(f"      ❌ 无法找到有效的卫生间中心点")
            return np.zeros((h, w), dtype=bool)
    
    print(f"      ✅ 使用中心点: ({center_x}, {center_y})")
    
    # 使用泛洪算法找到包含中心点的连通区域
    def flood_fill_bathroom(start_x, start_y):
        """找到包含起始点的完整卫生间区域"""
        visited = np.zeros((h, w), dtype=bool)
        room_mask = np.zeros((h, w), dtype=bool)
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            if (x < 0 or x >= w or y < 0 or y >= h or 
                visited[y, x] or floorplan[y, x] in [9, 10]):
                continue
            
            visited[y, x] = True
            room_mask[y, x] = True
            
            # 添加相邻的非墙壁像素
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((x + dx, y + dy))
        
        return room_mask
    
    # 使用泛洪算法获取完整的房间区域
    full_room_mask = flood_fill_bathroom(center_x, center_y)
    
    if np.sum(full_room_mask) == 0:
        print(f"      ❌ 泛洪算法未找到有效区域")
        return np.zeros((h, w), dtype=bool)
    
    # 计算泛洪得到的区域面积
    room_area = np.sum(full_room_mask)
    total_area = h * w
    room_ratio = room_area / total_area
    
    print(f"      📏 泛洪区域面积: {room_area} 像素 ({room_ratio:.1%})")
    
    # 如果泛洪区域过大，则创建更小的矩形区域
    if room_ratio > 0.06:  # 卫生间通常不超过6%
        print(f"      🎯 区域过大，创建适合的矩形卫生间")
        
        # 基于总面积计算合适的卫生间尺寸
        target_area = total_area * 0.015  # 目标1.5%的面积（更合理）
        target_size = int(np.sqrt(target_area))
        
        # 限制尺寸范围（针对大图像调整）
        min_size = max(15, min(h, w) // 25)  # 最小尺寸（更小）
        max_size = min(60, min(h, w) // 8)   # 最大尺寸（更小且有绝对上限）
        target_size = max(min_size, min(target_size, max_size))
        
        print(f"      🎯 目标卫生间尺寸: {target_size}x{target_size} 像素")
        
        # 创建卫生间掩码
        bathroom_mask = np.zeros((h, w), dtype=bool)
        
        # 计算矩形边界（以中心点为中心的正方形）
        half_size = target_size // 2
        
        # 确保边界在图像范围内
        left = max(0, center_x - half_size)
        right = min(w, center_x + half_size)
        top = max(0, center_y - half_size)
        bottom = min(h, center_y + half_size)
        
        # 调整边界，尽量保持正方形
        width = right - left
        height = bottom - top
        
        if width < target_size and right < w:
            right = min(w, right + (target_size - width))
        if height < target_size and bottom < h:
            bottom = min(h, bottom + (target_size - height))
        
        # 填充矩形区域，但只包含非墙壁的像素
        for y in range(top, bottom):
            for x in range(left, right):
                if floorplan[y, x] not in [9, 10]:  # 非墙壁
                    bathroom_mask[y, x] = True
        
        result_area = np.sum(bathroom_mask)
        print(f"      ✅ 创建矩形卫生间: {result_area} 像素")
        
        return bathroom_mask
    else:
        print(f"      ✅ 使用泛洪区域作为卫生间")
        return full_room_mask


def enhance_kitchen_detection(floorplan, ocr_results):
    """智能厨房检测：优先使用OCR，确保只识别一个厨房，形成规则的矩形区域"""
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # 首先检查OCR是否检测到厨房
    kitchen_ocr_items = []
    if ocr_results:
        for ocr_item in ocr_results:
            text = ocr_item['text'].lower()
            text_stripped = text.strip()
            
            # 定义关键词：单字符简写（完全匹配或开头匹配）+ 多字符关键词（包含匹配）
            single_char_keywords = ['厨']  # 中文简写
            multi_char_keywords = ['厨房', 'kitchen', 'cook', '烹饪', 'cooking']
            
            # 检查是否为单字符简写（完全匹配或以该字符开头，如"厨A"、"厨B"）
            is_single_char_match = (text_stripped in single_char_keywords or 
                                  any(text_stripped.startswith(char) for char in single_char_keywords))
            
            # 检查是否包含多字符关键词
            is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
            
            if is_single_char_match or is_multi_char_match:
                kitchen_ocr_items.append(ocr_item)
                print(f"🍳 OCR检测到厨房文字: '{ocr_item['text']}' (置信度: {ocr_item['confidence']:.3f})")
    
    # 如果OCR检测到厨房，优先使用OCR结果
    if kitchen_ocr_items:
        print("✅ 使用OCR检测的厨房位置")
        
        # 如果有多个厨房OCR结果，选择置信度最高的
        best_kitchen = max(kitchen_ocr_items, key=lambda x: x['confidence'])
        x, y, w, h = best_kitchen['bbox']
        center_x = x + w // 2
        center_y = y + h // 2
        
        print(f"   📍 选择最可靠的厨房: '{best_kitchen['text']}' (置信度: {best_kitchen['confidence']:.3f})")
        print(f"   🎯 厨房中心位置: ({center_x}, {center_y})")
        
        # 从OCR中心点生成规则的厨房区域（使用多边形模式）
        kitchen_mask = create_regular_kitchen_area(enhanced, center_x, center_y, h, w, mode='polygon')
        enhanced[kitchen_mask] = 7  # 厨房标签
        
        kitchen_pixels = np.sum(kitchen_mask)
        print(f"   ✅ 生成规则厨房区域: {kitchen_pixels} 像素")
        
        return enhanced
    
    # 如果OCR没有检测到厨房，使用空间分析（限制只识别一个）
    print("📍 OCR未检测到厨房，使用空间分析（限制识别一个厨房）")
    
    # 查找客厅/餐厅区域
    living_dining_mask = (enhanced == 3)
    
    if np.sum(living_dining_mask) == 0:
        print("❌ 未发现客厅/餐厅/厨房区域")
        return enhanced
    
    try:
        # 连通组件分析
        labeled_regions, num_regions = simple_connected_components(living_dining_mask)
        
        print(f"🔍 发现 {num_regions} 个客厅/餐厅区域")
        
        region_stats = []
        
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_area = np.sum(region_mask)
            
            # 获取区域边界
            region_coords = np.where(region_mask)
            if len(region_coords[0]) == 0:
                continue
                
            min_y, max_y = np.min(region_coords[0]), np.max(region_coords[0])
            min_x, max_x = np.min(region_coords[1]), np.max(region_coords[1])
            region_height = max_y - min_y + 1
            region_width = max_x - min_x + 1
            
            # 计算区域特征
            aspect_ratio = max(region_width, region_height) / min(region_width, region_height)
            density = region_area / (region_width * region_height)
            relative_area = region_area / (h * w)
            
            region_stats.append({
                'id': region_id,
                'mask': region_mask,
                'area': region_area,
                'relative_area': relative_area,
                'aspect_ratio': aspect_ratio,
                'density': density,
                'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
                'bbox': (min_x, min_y, max_x, max_y)
            })
        
        # 厨房选择策略：选择合适的区域作为厨房
        kitchen_candidates = []
        
        for stats in region_stats:
            print(f"   区域{stats['id']}: 面积={stats['relative_area']:.3f}, 长宽比={stats['aspect_ratio']:.2f}, 密度={stats['density']:.2f}")
            
            # 更严格的厨房候选条件：
            # 1. 面积要合适（不能太小也不能太大）
            # 2. 形状要相对规则
            # 3. 密度要合理
            # 4. 绝对面积要足够大
            absolute_area = stats['area']
            
            if (0.03 < stats['relative_area'] < 0.15 and  # 面积在3%-15%之间
                stats['aspect_ratio'] < 2.5 and          # 不太狭长  
                stats['density'] > 0.6 and               # 密度较高
                absolute_area > 500):                     # 绝对面积大于500像素
                
                kitchen_candidates.append(stats)
                print(f"      ✅ 厨房候选区域 (绝对面积: {absolute_area})")
            else:
                reasons = []
                if stats['relative_area'] <= 0.03:
                    reasons.append("面积太小")
                elif stats['relative_area'] >= 0.15:
                    reasons.append("面积太大")
                if stats['aspect_ratio'] >= 2.5:
                    reasons.append("形状狭长")
                if stats['density'] <= 0.6:
                    reasons.append("密度低")
                if absolute_area <= 500:
                    reasons.append("绝对面积不足")
                print(f"      ❌ 不符合厨房特征: {', '.join(reasons)}")
        
        # 如果有候选区域，选择最合适的一个作为厨房
        if kitchen_candidates:
            # 按面积和密度的综合评分排序
            def kitchen_score(stats):
                # 面积适中的得分更高，密度高的得分更高
                area_score = 1.0 - abs(stats['relative_area'] - 0.08) / 0.08
                density_score = stats['density']
                shape_score = 1.0 / stats['aspect_ratio']  # 越接近正方形得分越高
                return area_score * 0.4 + density_score * 0.4 + shape_score * 0.2
            
            kitchen_candidates.sort(key=kitchen_score, reverse=True)
            chosen_kitchen = kitchen_candidates[0]
            
            print(f"   🎯 选择区域{chosen_kitchen['id']}作为厨房")
            print(f"      面积: {chosen_kitchen['relative_area']:.3f}, 绝对面积: {chosen_kitchen['area']}")
            print(f"      长宽比: {chosen_kitchen['aspect_ratio']:.2f}, 密度: {chosen_kitchen['density']:.2f}")
            
            # 从区域中心生成规则的厨房区域（使用多边形模式）
            center_x, center_y = chosen_kitchen['center']
            kitchen_mask = create_regular_kitchen_area(enhanced, center_x, center_y, h, w, mode='polygon')
            
            if np.sum(kitchen_mask) > 0:
                enhanced[kitchen_mask] = 7  # 厨房标签
                kitchen_pixels = np.sum(kitchen_mask)
                print(f"   ✅ 生成规则厨房区域: {kitchen_pixels} 像素")
            else:
                print(f"   ❌ 无法在该区域生成有效的厨房")
        else:
            print("   ⚠️ 未找到符合条件的厨房候选区域")
                
    except Exception as e:
        print(f"⚠️ 空间分析出错: {e}")
        import traceback
        traceback.print_exc()
    
    return enhanced


def create_regular_kitchen_area(floorplan, center_x, center_y, img_h, img_w, mode='rect'):
    """从中心点创建厨房掩码，支持矩形和多边形两种模式。

    Args:
        floorplan: 分割图数组。
        center_x, center_y: 厨房中心点。
        img_h, img_w: 图像尺寸（兼容旧接口）。
        mode: 'rect' 使用矩形生成逻辑（默认，保持向后兼容），
              'polygon' 使用轮廓多边形生成掩码。
    """
    h, w = floorplan.shape
    
    # 确保坐标为整数类型
    center_x = int(center_x)
    center_y = int(center_y)

    print(f"      🏠 智能生成厨房区域: 中心({center_x}, {center_y}) 模式={mode}")

    # 首先检查中心点是否在有效区域（非墙壁）
    if floorplan[center_y, center_x] in [9, 10]:
        print(f"      ⚠️ 中心点在墙壁上，寻找附近的有效区域")
        found_valid = False
        for radius in range(1, 10):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    new_y, new_x = center_y + dy, center_x + dx
                    if (0 <= new_y < h and 0 <= new_x < w and
                        floorplan[new_y, new_x] not in [9, 10]):
                        center_x, center_y = new_x, new_y
                        found_valid = True
                        break
                if found_valid:
                    break
            if found_valid:
                break

        if not found_valid:
            print(f"      ❌ 无法找到有效的厨房中心点")
            return np.zeros((h, w), dtype=bool)

    print(f"      ✅ 使用中心点: ({center_x}, {center_y})")

    if mode == 'polygon':
        # 使用分割图获取连通区域轮廓
        room_label = floorplan[center_y, center_x]
        room_mask = (floorplan == room_label).astype(np.uint8)
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        selected_contour = None
        for cnt in contours:
            # 确保坐标为整数类型
            test_point = (int(center_x), int(center_y))
            if cv2.pointPolygonTest(cnt, test_point, False) >= 0:
                selected_contour = cnt
                break

        if selected_contour is None:
            print(f"      ❌ 未找到包含中心点的连通区域")
            return np.zeros((h, w), dtype=bool)

        # 多边形近似，保留凹凸结构
        epsilon = 0.01 * cv2.arcLength(selected_contour, True)
        approx = cv2.approxPolyDP(selected_contour, epsilon, True)

        area = cv2.contourArea(approx)
        x, y, bw, bh = cv2.boundingRect(approx)
        print(f"      📏 连通区域面积: {int(area)} 像素 ({area/(h*w):.1%})")
        print(f"      📐 轮廓边界: ({x},{y}) 到 ({x + bw},{y + bh}), 尺寸{bw}x{bh}")

        # 根据多边形生成厨房掩码
        kitchen_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(kitchen_mask, [approx], 1)
        kitchen_mask = kitchen_mask.astype(bool)

        valid_pixels = int(np.sum(kitchen_mask))
        print(f"      ✅ 厨房掩码生成完成: 有效像素 {valid_pixels}")
        # ===== 面积回退策略：若分割得到的厨房区域过小，则进行规则化扩展 =====
        total_area = h * w
        area_ratio = valid_pixels / total_area if total_area > 0 else 0
        min_polygon_ratio = 0.005   # 0.5% 以下视为异常小厨房
        target_min_ratio = 0.02     # 期望至少达到 2%
        target_pref_ratio = 0.035   # 目标 3.5%（处于 2-6% 合理区间内）

        if area_ratio < min_polygon_ratio:
            print(f"      ⚠️ 厨房多边形区域过小({area_ratio:.2%} < {min_polygon_ratio:.2%})，启动面积回退策略 -> 规则矩形扩展")
            target_area = total_area * target_pref_ratio
            target_size = int(min(max(np.sqrt(target_area), 24), min(h, w) / 3))
            half_size = target_size // 2
            left = max(0, center_x - half_size)
            right = min(w, center_x + half_size)
            top = max(0, center_y - half_size)
            bottom = min(h, center_y + half_size)
            # 若墙体太多逐步收缩
            for _ in range(5):
                region = floorplan[top:bottom, left:right]
                region_area = max(1, (right - left) * (bottom - top))
                wall_ratio = np.sum(np.isin(region, [9,10])) / region_area
                if wall_ratio <= 0.15:
                    break
                shrink_x = max(1, int((right - left) * 0.05))
                shrink_y = max(1, int((bottom - top) * 0.05))
                left += shrink_x; right -= shrink_x; top += shrink_y; bottom -= shrink_y
                left = max(0, left); top = max(0, top)
                right = max(left + 1, right); bottom = max(top + 1, bottom)
            rect_mask = np.zeros((h, w), dtype=bool)
            rect_mask[top:bottom, left:right] = True
            # 去除墙体像素，防止跨墙
            wall_mask = np.isin(floorplan, [9,10])
            if wall_mask.any():
                rect_mask[wall_mask] = False
            rect_pixels = int(rect_mask.sum())
            if rect_pixels / total_area < target_min_ratio:
                print(f"      🔄 回退矩形仍偏小({rect_pixels/total_area:.2%})，尝试膨胀填充")
                import cv2 as _cv2
                kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (5,5))
                temp = rect_mask.astype(np.uint8)
                for _ in range(4):
                    temp = _cv2.dilate(temp, kernel, iterations=1)
                    temp[np.isin(floorplan, [9,10])] = 0
                    if temp.sum() / total_area >= target_min_ratio:
                        break
                rect_mask = temp.astype(bool)
                rect_pixels = int(rect_mask.sum())
            # 仅保留与中心点连通的部分，避免跨墙越界
            if not rect_mask[center_y, center_x]:
                # 如果中心点被墙体剥离，尝试在邻域找一个在矩形内的有效点
                found=False
                for r in range(1,8):
                    for dy in range(-r,r+1):
                        for dx in range(-r,r+1):
                            ny=center_y+dy; nx=center_x+dx
                            if 0<=ny<h and 0<=nx<w and rect_mask[ny,nx]:
                                center_y, center_x = ny, nx; found=True; break
                        if found: break
                    if found: break
            from collections import deque as _deque
            visited = np.zeros_like(rect_mask, dtype=bool)
            if rect_mask[center_y, center_x]:
                q=_deque([(center_x, center_y)])
                visited[center_y, center_x]=True
                while q:
                    cx, cy = q.popleft()
                    for nx in (cx-1,cx,cx+1):
                        for ny in (cy-1,cy,cy+1):
                            if nx==cx and ny==cy: continue
                            if 0<=nx<w and 0<=ny<h and not visited[ny,nx] and rect_mask[ny,nx]:
                                visited[ny,nx]=True
                                q.append((nx,ny))
                disconnected = rect_mask & (~visited)
                disconnected_pixels = int(disconnected.sum())
                if disconnected_pixels>0:
                    print(f"      🔧 去除跨墙/不连通部分: {disconnected_pixels} 像素")
                rect_mask = visited
                rect_pixels = int(rect_mask.sum())
            print(f"      ✅ 面积回退后厨房区域: {rect_pixels} 像素 ({rect_pixels/total_area:.2%})")
            kitchen_mask = rect_mask
            valid_pixels = rect_pixels
        if valid_pixels / total_area < 0.005:
            print(f"      ❗ 仍检测到异常小厨房区域({valid_pixels/total_area:.2%})，建议检查模型对标签7的分割输出")
        return kitchen_mask

    # ======= 矩形模式 =======
    total_area = h * w
    # 厨房应该占总面积的2-6%，这是比较合理的范围
    target_area = total_area * 0.04  # 目标4%
    target_size = int(np.sqrt(target_area))

    # 设置尺寸限制：最小20像素，最大125像素
    min_size = 20
    max_size = min(125, min(h//4, w//4))  # 不超过图像尺寸的1/4
    target_size = max(min_size, min(target_size, max_size))

    print(f"      📏 发现房间区域: {total_area} 像素")
    print(f"      📐 房间边界: (0,0) 到 ({w-1},{h-1}), 尺寸{w}x{h}")
    print(f"      🎯 目标厨房尺寸: {target_size}x{target_size}")

    # 创建以中心点为中心的正方形厨房区域
    half_size = target_size // 2

    # 确保厨房区域在图像边界内
    kitchen_left = max(0, center_x - half_size)
    kitchen_right = min(w, center_x + half_size)
    kitchen_top = max(0, center_y - half_size)
    kitchen_bottom = min(h, center_y + half_size)

    # 调整尺寸确保是正方形（在图像边界内）
    kitchen_width = kitchen_right - kitchen_left
    kitchen_height = kitchen_bottom - kitchen_top

    # 如果不是正方形，调整到较小的尺寸
    if kitchen_width != kitchen_height:
        actual_size = min(kitchen_width, kitchen_height)
        half_actual = actual_size // 2

        # 重新计算边界，确保是正方形
        kitchen_left = max(0, center_x - half_actual)
        kitchen_right = min(w, center_x + half_actual)
        kitchen_top = max(0, center_y - half_actual)
        kitchen_bottom = min(h, center_y + half_actual)

    print(f"      ✅ 厨房区域生成完成:")
    print(f"         边界: ({kitchen_left},{kitchen_top}) 到 ({kitchen_right},{kitchen_bottom})")
    print(f"         尺寸: {kitchen_right-kitchen_left}x{kitchen_bottom-kitchen_top}")

    # 创建厨房掩码
    kitchen_mask = np.zeros((h, w), dtype=bool)
    kitchen_mask[kitchen_top:kitchen_bottom, kitchen_left:kitchen_right] = True

    valid_pixels = np.sum(kitchen_mask)
    print(f"         有效像素: {valid_pixels}")

    return kitchen_mask


def enhance_living_room_detection(floorplan, ocr_results):
    """
    增强客厅检测 - 优先使用OCR，然后使用空间分析
    """
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # 首先检查OCR是否检测到客厅
    living_room_ocr_items = []
    
    for ocr_item in ocr_results:
        text = ocr_item['text'].lower()
        text_stripped = text.strip()
        
        # 定义关键词：单字符简写（完全匹配或开头匹配）+ 多字符关键词（包含匹配）
        single_char_keywords = ['厅']  # 中文简写
        multi_char_keywords = ['客厅', '起居室', '大厅', '客餐厅', 'living', 'livingroom']
        
        # 检查是否为单字符简写（完全匹配或以该字符开头）
        is_single_char_match = (text_stripped in single_char_keywords or 
                              any(text_stripped.startswith(char) for char in single_char_keywords))
        
        # 检查是否包含多字符关键词
        is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
        
        if is_single_char_match or is_multi_char_match:
            living_room_ocr_items.append(ocr_item)
            print(f"🏠 OCR检测到客厅文字: '{ocr_item['text']}' (置信度: {ocr_item['confidence']:.3f})")
    
    # 如果OCR检测到客厅，优先使用OCR结果
    if living_room_ocr_items:
        print("✅ 使用OCR检测的客厅位置")
        for i, ocr_item in enumerate(living_room_ocr_items):
            print(f"   📍 处理客厅 {i+1}: '{ocr_item['text']}' (置信度: {ocr_item['confidence']:.3f})")
            
            # 获取OCR文字中心的512x512坐标
            x, y, w, h = ocr_item['bbox']
            center_x_512 = x + w // 2
            center_y_512 = y + h // 2
            print(f"   🎯 客厅中心位置: ({center_x_512}, {center_y_512})")
            
            # 在该位置创建客厅区域
            living_room_mask = create_regular_living_room_area(enhanced, center_x_512, center_y_512, h, w)
            if living_room_mask is not None:
                # 设置为客厅类别（3）
                enhanced[living_room_mask] = 3
                living_room_pixels = np.sum(living_room_mask)
                print(f"   ✅ 生成规则客厅区域 {i+1}: {living_room_pixels} 像素")
    
    # 如果OCR没有检测到客厅，使用空间分析
    else:
        print("📍 OCR未检测到客厅，使用空间分析方法")
        
        try:
            # 查找客厅区域（通常是最大的房间）
            potential_living_mask = (enhanced == 3) | (enhanced == 1)  # 客厅或未分类区域
            labeled_regions, num_regions = label(potential_living_mask, connectivity=2, return_num=True)
            
            if num_regions == 0:
                print("❌ 未发现客厅区域")
                return enhanced
            else:
                print(f"🔍 发现 {num_regions} 个客厅候选区域")
            
            # 统计各区域特征
            region_stats = []
            total_pixels = h * w
            
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_regions == region_id)
                area = np.sum(region_mask)
                area_ratio = area / total_pixels
                
                # 计算边界框和紧凑度
                y_coords, x_coords = np.where(region_mask)
                if len(y_coords) > 0:
                    min_x, max_x = x_coords.min(), x_coords.max()
                    min_y, max_y = y_coords.min(), y_coords.max()
                    bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                    compactness = area / bbox_area if bbox_area > 0 else 0
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    
                    region_stats.append({
                        'id': region_id,
                        'area': area,
                        'area_ratio': area_ratio,
                        'compactness': compactness,
                        'center': (center_x, center_y),
                        'bbox': (min_x, min_y, max_x, max_y)
                    })
            
            # 筛选符合客厅特征的区域
            living_room_candidates = []
            for stat in region_stats:
                # 客厅特征：面积较大（通常>8%），形状相对规整
                if (stat['area_ratio'] >= 0.08 and  # 客厅面积通常较大
                    stat['compactness'] >= 0.2):     # 形状相对规整
                    living_room_candidates.append(stat)
                    print(f"   🏠 发现客厅候选区域: 面积比例={stat['area_ratio']:.3f}, 紧凑度={stat['compactness']:.3f}")
            
            # 选择最大的区域作为客厅（客厅通常是最大的房间）
            if living_room_candidates:
                # 按面积排序，选择最大的
                living_room_candidates.sort(key=lambda x: x['area'], reverse=True)
                best_living_room = living_room_candidates[0]
                
                print(f"   🎯 选择最大区域作为客厅: 面积比例={best_living_room['area_ratio']:.3f}")
                
                # 在该区域创建客厅
                center_x, center_y = best_living_room['center']
                living_room_mask = create_regular_living_room_area(enhanced, center_x, center_y, h, w)
                if living_room_mask is not None:
                    enhanced[living_room_mask] = 3
                    living_room_pixels = np.sum(living_room_mask)
                    print(f"   ✅ 生成规则客厅区域: {living_room_pixels} 像素")
                else:
                    print(f"   ❌ 无法在该区域生成有效的客厅")
            else:
                print("   ⚠️ 未找到符合条件的客厅候选区域")
                    
        except Exception as e:
            print(f"⚠️ 客厅空间分析出错: {e}")
            import traceback
            traceback.print_exc()
    
    return enhanced


def create_regular_living_room_area(floorplan, center_x, center_y, img_h, img_w):
    """从中心点创建规则的客厅区域，严格限制在房间边界内"""
    h, w = floorplan.shape
    
    print(f"      🏠 智能生成客厅区域: 中心({center_x}, {center_y})")
    
    # 首先检查中心点是否在有效区域（非墙壁）
    if floorplan[center_y, center_x] in [9, 10]:
        print(f"      ⚠️ 中心点在墙壁上，寻找附近的有效区域")
        # 寻找附近的非墙壁区域
        found_valid = False
        for radius in range(1, 12):  # 扩大搜索范围（客厅较大）
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = center_y + dy, center_x + dx
                    if (0 <= ny < h and 0 <= nx < w and 
                        floorplan[ny, nx] not in [9, 10]):
                        center_x, center_y = nx, ny
                        found_valid = True
                        print(f"      ✅ 使用中心点: ({center_x}, {center_y})")
                        break
                if found_valid:
                    break
            if found_valid:
                break
        
        if not found_valid:
            print(f"      ❌ 无法找到有效的中心点")
            return None
    else:
        print(f"      ✅ 使用中心点: ({center_x}, {center_y})")
    
    # 使用泛洪算法找到连通区域
    try:
        from skimage.segmentation import flood
        # 泛洪填充，找到连通的房间区域
        room_mask = flood(floorplan, (center_y, center_x), tolerance=0)
        room_pixels = np.sum(room_mask)
        room_ratio = room_pixels / (h * w)
        
        print(f"      📏 泛洪区域面积: {room_pixels} 像素 ({room_ratio:.1%})")
        
        # 如果泛洪区域合理，直接使用
        if 0.05 <= room_ratio <= 0.25:  # 客厅面积限制在25%以内，防止过大重叠
            print(f"      ✅ 使用泛洪区域作为客厅")
            return room_mask
        elif room_ratio > 0.25:
            print(f"      ⚠️ 泛洪区域过大({room_ratio:.1%})，使用矩形区域")
    except:
        print(f"      ⚠️ 泛洪算法失败，使用矩形区域")
    
    # 如果泛洪失败，创建矩形区域
    # 分析周围区域，找到房间边界
    room_mask = np.zeros((h, w), dtype=bool)
    
    # 从中心点向四个方向扩展，找到墙壁边界
    # 向左扩展
    min_x = center_x
    for x in range(center_x, -1, -1):
        if floorplan[center_y, x] in [9, 10]:  # 遇到墙壁停止
            min_x = x + 1
            break
        min_x = x
    
    # 向右扩展  
    max_x = center_x
    for x in range(center_x, w):
        if floorplan[center_y, x] in [9, 10]:  # 遇到墙壁停止
            max_x = x - 1
            break
        max_x = x
    
    # 向上扩展
    min_y = center_y
    for y in range(center_y, -1, -1):
        if floorplan[y, center_x] in [9, 10]:  # 遇到墙壁停止
            min_y = y + 1
            break
        min_y = y
    
    # 向下扩展
    max_y = center_y
    for y in range(center_y, h):
        if floorplan[y, center_x] in [9, 10]:  # 遇到墙壁停止
            max_y = y - 1
            break
        max_y = y
    
    room_width = max_x - min_x + 1
    room_height = max_y - min_y + 1
    
    print(f"      📐 房间边界: ({min_x},{min_y}) 到 ({max_x},{max_y}), 尺寸{room_width}x{room_height}")
    
    # 根据房间大小确定客厅尺寸（客厅通常较大，使用更大比例）
    max_living_width = int(room_width * 0.9)  # 客厅可以占用更大比例
    max_living_height = int(room_height * 0.9)
    
    # 计算理想的客厅尺寸
    total_area = h * w
    target_area = min(total_area * 0.15, room_pixels * 0.8)  # 客厅最多占总面积15%或房间80%
    target_size = int(np.sqrt(target_area))
    
    # 限制客厅大小
    min_size = 30  # 客厅最小尺寸较大
    target_size = max(min_size, min(target_size, min(max_living_width, max_living_height)))
    
    print(f"      📏 目标客厅尺寸: {target_size}x{target_size}")
    
    # 在房间内创建以中心点为中心的客厅区域
    half_size = target_size // 2
    
    living_left = max(min_x, center_x - half_size)
    living_right = min(max_x + 1, center_x + half_size)
    living_top = max(min_y, center_y - half_size)
    living_bottom = min(max_y + 1, center_y + half_size)
    
    # 调整尺寸以达到目标大小
    living_width = living_right - living_left
    living_height = living_bottom - living_top
    
    # 尝试扩展到目标尺寸
    if living_width < target_size:
        # 尝试扩展宽度
        needed = target_size - living_width
        if living_left - needed//2 >= min_x:
            living_left -= needed//2
        elif living_right + needed//2 <= max_x + 1:
            living_right += needed//2
    
    if living_height < target_size:
        # 尝试扩展高度
        needed = target_size - living_height
        if living_top - needed//2 >= min_y:
            living_top -= needed//2
        elif living_bottom + needed//2 <= max_y + 1:
            living_bottom += needed//2
    
    # 创建客厅掩码，只在房间区域内
    living_mask = np.zeros((h, w), dtype=bool)
    
    for y in range(living_top, living_bottom):
        for x in range(living_left, living_right):
            # 如果room_mask生成失败，直接创建客厅区域（避免墙壁）
            if room_mask is not None and room_mask[y, x]:  
                living_mask[y, x] = True
            elif room_mask is None and floorplan[y, x] not in [9, 10]:  # 备用方案：非墙壁即可
                living_mask[y, x] = True
    
    actual_width = living_right - living_left
    actual_height = living_bottom - living_top
    actual_pixels = np.sum(living_mask)
    
    # 如果生成的客厅区域过小，使用简单的矩形区域
    if actual_pixels < 100:  # 如果客厅区域太小
        print(f"      ⚠️ 客厅区域过小({actual_pixels}像素)，使用扩展矩形回退")
        living_mask.fill(False)
        # 基于已检测房间边界扩大: 取 bounding box 60% 尺寸的方形
        box_w = max_x - min_x + 1
        box_h = max_y - min_y + 1
        side = int(min(max(box_w, box_h), max( min(box_w, box_h) * 1.2, target_size*1.2 )))
        side = min(side, int(min(h, w)*0.9))
        half = side//2
        living_left = max(0, center_x - half)
        living_right = min(w, center_x + half)
        living_top = max(0, center_y - half)
        living_bottom = min(h, center_y + half)
        for y in range(living_top, living_bottom):
            for x in range(living_left, living_right):
                if floorplan[y, x] not in [9,10]:
                    living_mask[y,x] = True
        actual_pixels = living_mask.sum()
        print(f"      ✅ 回退扩展后客厅像素: {actual_pixels}")
    
    print(f"      ✅ 客厅区域生成完成:")
    print(f"         边界: ({living_left},{living_top}) 到 ({living_right},{living_bottom})")
    print(f"         尺寸: {actual_width}x{actual_height}")
    print(f"         有效像素: {actual_pixels}")
    
    return living_mask


def ind2rgb(ind_im, enable_closet=True):
        # Use the appropriate color map based on closet setting
        if enable_closet:
            color_map = floorplan_fuse_map_figure
        else:
            # Create a modified map without closet
            color_map = floorplan_fuse_map_figure.copy()
            color_map[1] = color_map[3]  # Map closet to living room color
        
        rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

        for i, rgb in color_map.items():
                rgb_im[(ind_im==i)] = rgb

        return rgb_im

def main(args):
        # Default behavior: closet is disabled unless explicitly enabled
        enable_closet = args.enable_closet
        # Support legacy --disable_closet flag for backward compatibility
        if args.disable_closet:
            enable_closet = False
        set_closet_enabled(enable_closet)

        # Handle image path - support both positional and --im_path argument
        image_path = args.im_path if args.im_path else args.image_path

        # load input
        im = imread(image_path, mode='RGB')
        # Keep original size for better OCR
        original_im = im.copy()
        
        print(f"🖼️ 图像处理流程详细分析:")
        print(f"   📏 原始图像尺寸: {original_im.shape[1]} x {original_im.shape[0]} (宽x高)")
        
        # Resize image for network inference
        im = imresize(im, (512, 512))
        print(f"   🔄 神经网络输入: 512 x 512 (固定尺寸)")
        print(f"   💡 为什么要转换到512x512？")
        print(f"      - DeepFloorplan神经网络模型训练时使用的是512x512输入")
        print(f"      - 所有的神经网络推理都在512x512坐标系中进行")
        print(f"      - 最终结果需要转换回原始图像尺寸用于显示")
        
        # For OCR, use larger, enhanced image
        from PIL import Image, ImageEnhance
        ocr_img = Image.fromarray(original_im)
        # Enlarge for better OCR
        ocr_img = ocr_img.resize((ocr_img.width * 2, ocr_img.height * 2), Image.LANCZOS)
        print(f"   🔍 OCR处理图像: {ocr_img.width} x {ocr_img.height} (放大2倍)")
        print(f"   💡 为什么OCR要放大2倍？")
        print(f"      - 提高OCR识别小文字的准确性")
        print(f"      - 增强文字的清晰度和对比度")
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(ocr_img)
        ocr_img = enhancer.enhance(2.5)
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(ocr_img)
        ocr_img = enhancer.enhance(2.0)
        ocr_im = np.array(ocr_img)
        
        print(f"   📊 坐标转换关系:")
        print(f"      原始图像 -> 512x512: 缩放比例 X={512/original_im.shape[1]:.3f}, Y={512/original_im.shape[0]:.3f}")
        print(f"      512x512 -> 原始图像: 缩放比例 X={original_im.shape[1]/512:.3f}, Y={original_im.shape[0]/512:.3f}")
        print(f"      ⚠️ 关键: X比例={original_im.shape[1]/512:.3f} 就是您看到的1.131!")
        print(f"      📐 计算: {original_im.shape[1]} ÷ 512 = {original_im.shape[1]/512:.3f}")
        
        # Extract textual room labels using OCR with enhanced image
        ocr_results = extract_room_text(ocr_im)
        # Scale OCR bounding boxes to match segmentation size (512x512)
        if ocr_results:
                scale_x = im.shape[1] / ocr_im.shape[1]
                scale_y = im.shape[0] / ocr_im.shape[0]
                print(f"   🔄 OCR坐标转换到512x512:")
                print(f"      OCR图像({ocr_im.shape[1]}x{ocr_im.shape[0]}) -> 512x512")
                print(f"      转换比例: X={scale_x:.3f}, Y={scale_y:.3f}")
                for item in ocr_results:
                        x, y, w, h = item['bbox']
                        x = int(x * scale_x)
                        y = int(y * scale_y)
                        w = int(w * scale_x)
                        h = int(h * scale_y)
                        item['bbox'] = (x, y, w, h)
        
        # Convert to float and normalize for network inference
        im = im.astype(np.float32) / 255.

        # 检测GPU可用性并配置TensorFlow
        gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0 if hasattr(tf.config, 'experimental') else False
        if not gpu_available:
            try:
                # TF 1.x的GPU检测方法
                from tensorflow.python.client import device_lib
                local_devices = device_lib.list_local_devices()
                gpu_available = any(device.device_type == 'GPU' for device in local_devices)
            except:
                gpu_available = False
        
        print(f"💻 设备状态: {'GPU可用' if gpu_available else 'CPU模式'}")
        
        # Create tensorflow session with optimized configuration
        config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
        )
        # Enable GPU memory growth to avoid allocation issues
        if gpu_available:
            config.gpu_options.allow_growth = True
            print("🚀 使用GPU加速")
        else:
            # Disable GPU if not available
            print("🔧 使用CPU计算")
            
        with tf.Session(config=config) as sess:
                
                # initialize
                sess.run(tf.group(tf.global_variables_initializer(),
                                        tf.local_variables_initializer()))

                # restore pretrained model
                saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
                saver.restore(sess, './pretrained/pretrained_r3d')

                # get default graph
                graph = tf.get_default_graph()

                # restore inputs & outpus tensor
                x = graph.get_tensor_by_name('inputs:0')
                room_type_logit = graph.get_tensor_by_name('Cast:0')
                room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

                # infer results
                [room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
                                                                feed_dict={x:im.reshape(1,512,512,3)})
                room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

                # merge results
                floorplan = room_type.copy()
                floorplan[room_boundary==1] = 9
                floorplan[room_boundary==2] = 10
                
                # Use OCR labels to refine room categories
                floorplan = fuse_ocr_and_segmentation(floorplan, ocr_results)
                
                # 智能厨房检测 - 只识别一个厨房，形成规则区域
                floorplan = enhance_kitchen_detection(floorplan, ocr_results)
                
                # 🚿 智能卫生间检测 - 精准识别卫生间，形成规则区域
                floorplan = enhance_bathroom_detection(floorplan, ocr_results)
                
                # 🏠 智能客厅检测 - 精准识别客厅，形成规则区域
                floorplan = enhance_living_room_detection(floorplan, ocr_results)
                
                # 获取厨房位置用于可视化标记
                kitchen_boxes = []
                if ocr_results:
                    for ocr_item in ocr_results:
                        text = ocr_item['text'].lower()
                        text_stripped = text.strip()
                        
                        # 厨房关键词匹配（与厨房检测函数保持一致）
                        single_char_keywords = ['厨']
                        multi_char_keywords = ['厨房', 'kitchen', 'cook', '烹饪', 'cooking']
                        
                        is_single_char_match = (text_stripped in single_char_keywords or 
                                              any(text_stripped.startswith(char) for char in single_char_keywords))
                        is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
                        
                        if is_single_char_match or is_multi_char_match:
                            x, y, w, h = ocr_item['bbox']
                            ocr_center_x = x + w // 2
                            ocr_center_y = y + h // 2
                            orig_center_x = int(ocr_center_x * original_im.shape[1] / 512)
                            orig_center_y = int(ocr_center_y * original_im.shape[0] / 512)
                            
                            kitchen_boxes.append({
                                'center': (ocr_center_x, ocr_center_y),
                                'original_center': (orig_center_x, orig_center_y),
                                'bbox': (x, y, w, h),
                                'text': ocr_item['text'],
                                'confidence': ocr_item['confidence']
                            })
                            # 只要第一个厨房
                            break
                
                # 🚿 获取卫生间位置用于可视化标记
                bathroom_boxes = []
                if ocr_results:
                    for ocr_item in ocr_results:
                        text = ocr_item['text'].lower()
                        text_stripped = text.strip()
                        
                        # 卫生间关键词匹配（与卫生间检测函数保持一致）
                        single_char_keywords = ['卫', '洗', '浴']
                        multi_char_keywords = ['卫生间', 'bathroom', 'toilet', 'wc', '厕所', 
                                             '浴室', 'shower', 'bath', '洗手间', '卫浴', 
                                             'restroom', 'washroom']
                        
                        is_single_char_match = (text_stripped in single_char_keywords or 
                                              any(text_stripped.startswith(char) for char in single_char_keywords))
                        is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
                        
                        if is_single_char_match or is_multi_char_match:
                            x, y, w, h = ocr_item['bbox']
                            ocr_center_x = x + w // 2
                            ocr_center_y = y + h // 2
                            orig_center_x = int(ocr_center_x * original_im.shape[1] / 512)
                            orig_center_y = int(ocr_center_y * original_im.shape[0] / 512)
                            
                            bathroom_boxes.append({
                                'center': (ocr_center_x, ocr_center_y),
                                'original_center': (orig_center_x, orig_center_y),
                                'bbox': (x, y, w, h),
                                'text': ocr_item['text'],
                                'confidence': ocr_item['confidence']
                            })
                
                # 🏠 获取客厅位置用于可视化标记
                living_room_boxes = []
                if ocr_results:
                    for ocr_item in ocr_results:
                        text = ocr_item['text'].lower()
                        text_stripped = text.strip()
                        
                        # 客厅关键词匹配（与客厅检测函数保持一致）
                        single_char_keywords = ['厅']
                        multi_char_keywords = ['客厅', '起居室', '大厅', '客餐厅', 'living', 'livingroom']
                        
                        is_single_char_match = (text_stripped in single_char_keywords or 
                                              any(text_stripped.startswith(char) for char in single_char_keywords))
                        is_multi_char_match = any(keyword in text for keyword in multi_char_keywords)
                        
                        if is_single_char_match or is_multi_char_match:
                            x, y, w, h = ocr_item['bbox']
                            ocr_center_x = x + w // 2
                            ocr_center_y = y + h // 2
                            orig_center_x = int(ocr_center_x * original_im.shape[1] / 512)
                            orig_center_y = int(ocr_center_y * original_im.shape[0] / 512)
                            
                            living_room_boxes.append({
                                'center': (ocr_center_x, ocr_center_y),
                                'original_center': (orig_center_x, orig_center_y),
                                'bbox': (x, y, w, h),
                                'text': ocr_item['text'],
                                'confidence': ocr_item['confidence']
                            })
                
                # Handle closet disable
                if not enable_closet:
                        floorplan[floorplan==1] = 0
                floorplan_rgb = ind2rgb(floorplan, enable_closet)
                
                # 🎯 添加厨房位置的红色标记和坐标网格
                if kitchen_boxes:
                    # 转换为原始图像尺寸
                    original_h, original_w = original_im.shape[:2]
                    floorplan_original_size = cv2.resize(floorplan_rgb, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    
                    # 添加更明显的坐标网格 (每25像素一条细线，每100像素一条粗线)
                    for x in range(0, original_w, 25):
                        thickness = 2 if x % 100 == 0 else 1
                        color = (0, 0, 255) if x % 100 == 0 else (128, 128, 128)  # 红色主线，灰色细线
                        cv2.line(floorplan_original_size, (x, 0), (x, original_h), color, thickness)
                    for y in range(0, original_h, 25):
                        thickness = 2 if y % 100 == 0 else 1
                        color = (0, 0, 255) if y % 100 == 0 else (128, 128, 128)  # 红色主线，灰色细线
                        cv2.line(floorplan_original_size, (0, y), (original_w, y), color, thickness)
                    
                    # 添加坐标标注 (每50像素)
                    for x in range(0, original_w, 50):
                        cv2.putText(floorplan_original_size, str(x), (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    for y in range(0, original_h, 50):
                        cv2.putText(floorplan_original_size, str(y), (5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 标记OCR识别的厨房文字中心位置
                    for kitchen_info in kitchen_boxes:
                        # OCR识别的厨房文字中心位置（绿色 - 这就是"厨房"两字的精确中心）
                        ocr_x, ocr_y = kitchen_info['original_center']
                        cv2.rectangle(floorplan_original_size, 
                                    (ocr_x-30, ocr_y-30), 
                                    (ocr_x+30, ocr_y+30), 
                                    (0, 255, 0), 4)
                        cv2.circle(floorplan_original_size, (ocr_x, ocr_y), 8, (0, 255, 0), -1)
                        
                        # OCR位置标注
                        ocr_text = f"OCR厨房文字中心({ocr_x},{ocr_y})"
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, ocr_text, 
                                  (ocr_x+35, ocr_y-20), 16, (0, 255, 0))
                        
                        print(f"🎯 OCR识别的厨房文字中心: 绿色框({ocr_x}, {ocr_y})")
                        print(f"� 这是'厨房'两个字的精确中心位置")
                        
                        # 添加图例说明
                        legend_y = 30
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "绿色=OCR识别厨房文字中心", 
                                  (10, legend_y), 16, (0, 255, 0))
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "这是'厨房'两字的精确位置", 
                                  (10, legend_y + 25), 16, (0, 255, 0))
                    
                    # 🚿 标记OCR识别的卫生间文字中心位置
                    bathroom_legend_offset = len(kitchen_boxes) * 50  # 根据厨房数量调整图例位置
                    for i, bathroom_info in enumerate(bathroom_boxes):
                        # OCR识别的卫生间文字中心位置（蓝色 - 这就是"卫生间"文字的精确中心）
                        ocr_x, ocr_y = bathroom_info['original_center']
                        cv2.rectangle(floorplan_original_size, 
                                    (ocr_x-25, ocr_y-25), 
                                    (ocr_x+25, ocr_y+25), 
                                    (255, 0, 0), 4)  # 蓝色矩形
                        cv2.circle(floorplan_original_size, (ocr_x, ocr_y), 6, (255, 0, 0), -1)  # 蓝色圆点
                        
                        # OCR位置标注
                        ocr_text = f"OCR卫生间{i+1}({ocr_x},{ocr_y})"
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, ocr_text, 
                                  (ocr_x+30, ocr_y-15), 16, (255, 0, 0))
                        
                        print(f"🚿 OCR识别的卫生间{i+1}文字中心: 蓝色框({ocr_x}, {ocr_y})")
                        print(f"🎯 这是'{bathroom_info['text']}'文字的精确中心位置")
                    
                    # 添加卫生间图例说明
                    if bathroom_boxes:
                        legend_y_bathroom = 30 + bathroom_legend_offset
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "蓝色=OCR识别卫生间文字中心", 
                                  (10, legend_y_bathroom), 16, (255, 0, 0))
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "精准定位卫生间位置", 
                                  (10, legend_y_bathroom + 25), 16, (255, 0, 0))
                    
                    # 🏠 标记OCR识别的客厅文字中心位置
                    living_room_legend_offset = len(kitchen_boxes) * 50 + len(bathroom_boxes) * 50  # 根据厨房和卫生间数量调整图例位置
                    for i, living_room_info in enumerate(living_room_boxes):
                        # OCR识别的客厅文字中心位置（橙色 - 这就是"客厅"文字的精确中心）
                        ocr_x, ocr_y = living_room_info['original_center']
                        cv2.rectangle(floorplan_original_size, 
                                    (ocr_x-25, ocr_y-25), 
                                    (ocr_x+25, ocr_y+25), 
                                    (0, 165, 255), 4)  # 橙色矩形
                        cv2.circle(floorplan_original_size, (ocr_x, ocr_y), 6, (0, 165, 255), -1)  # 橙色圆点
                        
                        # OCR位置标注
                        ocr_text = f"OCR客厅{i+1}({ocr_x},{ocr_y})"
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, ocr_text, 
                                  (ocr_x+30, ocr_y-15), 16, (0, 165, 255))
                        
                        print(f"🏠 OCR识别的客厅{i+1}文字中心: 橙色框({ocr_x}, {ocr_y})")
                        print(f"🎯 这是'{living_room_info['text']}'文字的精确中心位置")
                    
                    # 添加客厅图例说明
                    if living_room_boxes:
                        legend_y_living_room = 30 + living_room_legend_offset
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "橙色=OCR识别客厅文字中心", 
                                  (10, legend_y_living_room), 16, (0, 165, 255))
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "精准定位客厅位置", 
                                  (10, legend_y_living_room + 25), 16, (0, 165, 255))
                    
                    # 保存带标记的结果到output文件夹
                    output_dir = 'output'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    base_name = os.path.basename(image_path).replace('.jpg', '_rooms_marked.png').replace('.png', '_rooms_marked.png')
                    marked_filename = os.path.join(output_dir, base_name)
                    imsave(marked_filename, floorplan_original_size)
                    
                    # 打印总结信息
                    total_detections = len(kitchen_boxes) + len(bathroom_boxes) + len(living_room_boxes)
                    print(f"✅ 带标记的结果已保存: {marked_filename}")
                    print(f"🏠 检测摘要: {len(kitchen_boxes)}个厨房 + {len(bathroom_boxes)}个卫生间 + {len(living_room_boxes)}个客厅 = {total_detections}个房间")
                    
                    if kitchen_boxes:
                        print(f"🍳 厨房检测: 绿色标记")
                    if bathroom_boxes:
                        print(f"🚿 卫生间检测: 蓝色标记")
                    if living_room_boxes:
                        print(f"🏠 客厅检测: 橙色标记")
                else:
                    # 没有厨房时，也添加坐标网格和卫生间标记便于分析
                    original_h, original_w = original_im.shape[:2]
                    floorplan_original_size = cv2.resize(floorplan_rgb, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    
                    # 添加坐标网格
                    for x in range(0, original_w, 25):
                        thickness = 2 if x % 100 == 0 else 1
                        color = (0, 0, 255) if x % 100 == 0 else (128, 128, 128)
                        cv2.line(floorplan_original_size, (x, 0), (x, original_h), color, thickness)
                    for y in range(0, original_h, 25):
                        thickness = 2 if y % 100 == 0 else 1
                        color = (0, 0, 255) if y % 100 == 0 else (128, 128, 128)
                        cv2.line(floorplan_original_size, (0, y), (original_w, y), color, thickness)
                    
                    # 添加坐标标注
                    for x in range(0, original_w, 50):
                        cv2.putText(floorplan_original_size, str(x), (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    for y in range(0, original_h, 50):
                        cv2.putText(floorplan_original_size, str(y), (5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 🚿 即使没有厨房，也标记卫生间
                    for i, bathroom_info in enumerate(bathroom_boxes):
                        ocr_x, ocr_y = bathroom_info['original_center']
                        cv2.rectangle(floorplan_original_size, 
                                    (ocr_x-25, ocr_y-25), 
                                    (ocr_x+25, ocr_y+25), 
                                    (255, 0, 0), 4)  # 蓝色矩形
                        cv2.circle(floorplan_original_size, (ocr_x, ocr_y), 6, (255, 0, 0), -1)  # 蓝色圆点
                        
                        ocr_text = f"OCR卫生间{i+1}({ocr_x},{ocr_y})"
                        text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(floorplan_original_size,
                                    (ocr_x+28, ocr_y-30),
                                    (ocr_x+32+text_size[0], ocr_y-10),
                                    (255, 255, 255), -1)
                        cv2.putText(floorplan_original_size, ocr_text, 
                                  (ocr_x+30, ocr_y-15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # 添加图例
                    if bathroom_boxes:
                        floorplan_original_size = draw_chinese_text(floorplan_original_size, "蓝色=OCR识别卫生间文字中心", 
                                  (10, 30), 16, (255, 0, 0))
                    
                    # 保存结果到output文件夹
                    if bathroom_boxes:
                        output_dir = 'output'
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        base_name = os.path.basename(image_path).replace('.jpg', '_bathroom_marked.png').replace('.png', '_bathroom_marked.png')
                        marked_filename = os.path.join(output_dir, base_name)
                        imsave(marked_filename, floorplan_original_size)
                        print(f"✅ 带卫生间标记的结果已保存: {marked_filename}")
                        print(f"🚿 检测到 {len(bathroom_boxes)} 个卫生间")

                # plot results with coordinate axes
                plt.figure(figsize=(18, 8))
                plt.subplot(131)
                plt.imshow(im)
                plt.title('原始图片')
                plt.axis('on')  # 显示坐标轴
                plt.grid(True, alpha=0.3)
                
                plt.subplot(132)
                plt.imshow(floorplan_rgb/255.)
                plt.title('户型分析结果 (绿色=厨房)')
                plt.axis('on')  # 显示坐标轴
                plt.grid(True, alpha=0.3)
                
                # 第三个子图：显示带标记的结果
                if kitchen_boxes:
                    plt.subplot(133)
                    plt.imshow(floorplan_original_size)
                    plt.title('厨房标记结果 (红色=厨房位置)')
                    plt.axis('on')  # 显示坐标轴
                    plt.grid(True, alpha=0.3)
                    
                    # 在图上标注厨房坐标
                    for kitchen_info in kitchen_boxes:
                        orig_x, orig_y = kitchen_info['original_center']
                        plt.plot(orig_x, orig_y, 'ro', markersize=8, label=f"厨房({orig_x},{orig_y})")
                        plt.annotate(f"{kitchen_info['text']}\n({orig_x},{orig_y})", 
                                   (orig_x, orig_y), 
                                   xytext=(10, -10), 
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                                   fontsize=10, color='white')
                    
                    # 在图上标注客厅坐标
                    for living_room_info in living_room_boxes:
                        orig_x, orig_y = living_room_info['original_center']
                        plt.plot(orig_x, orig_y, 'o', color='orange', markersize=8, label=f"客厅({orig_x},{orig_y})")
                        plt.annotate(f"{living_room_info['text']}\n({orig_x},{orig_y})", 
                                   (orig_x, orig_y), 
                                   xytext=(10, 10), 
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                                   fontsize=10, color='white')
                    plt.legend()
                
                # Save result to output folder
                output_dir = 'output'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_name = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_result.png')
                plt.savefig(output_name, dpi=300, bbox_inches='tight')
                print(f"📸 结果已保存: {output_name}")
                
                plt.show()

if __name__ == '__main__':
        FLAGS, unparsed = parser.parse_known_args()
        main(FLAGS)
