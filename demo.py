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

tf.logging.set_verbosity(tf.logging.ERROR)  # 减少TensorFlow日志

# Configure Chinese font support for matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Disable TF 2.x behavior for compatibility
tf.disable_v2_behavior()

from PIL import Image
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
        if any(keyword in text for keyword in ['厨房', 'kitchen', 'cook', '烹饪']):
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


def enhance_kitchen_detection(floorplan, ocr_results):
    """智能厨房检测：优先使用OCR，确保只识别一个厨房，形成规则的矩形区域"""
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # 首先检查OCR是否检测到厨房
    kitchen_ocr_items = []
    if ocr_results:
        for ocr_item in ocr_results:
            text = ocr_item['text'].lower()
            if any(keyword in text for keyword in ['厨房', 'kitchen', 'cook', '烹饪']):
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
        
        # 从OCR中心点生成规则的厨房区域
        kitchen_mask = create_regular_kitchen_area(enhanced, center_x, center_y, h, w)
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
            
            # 从区域中心生成规则的厨房区域
            center_x, center_y = chosen_kitchen['center']
            kitchen_mask = create_regular_kitchen_area(enhanced, center_x, center_y, h, w)
            
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


def create_regular_kitchen_area(floorplan, center_x, center_y, img_h, img_w):
    """从中心点创建规则的矩形厨房区域，严格限制在房间边界内"""
    h, w = floorplan.shape
    
    print(f"      🏠 智能生成厨房区域: 中心({center_x}, {center_y})")
    
    # 首先检查中心点是否在有效区域（非墙壁）
    if floorplan[center_y, center_x] in [9, 10]:
        print(f"      ⚠️ 中心点在墙壁上，寻找附近的有效区域")
        # 寻找附近的非墙壁区域
        found_valid = False
        for radius in range(1, 10):
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
            print(f"      ❌ 无法找到有效的厨房中心点")
            return np.zeros((h, w), dtype=bool)
    
    print(f"      ✅ 使用中心点: ({center_x}, {center_y})")
    
    # 使用泛洪算法找到包含中心点的连通区域
    def flood_fill_room(start_x, start_y):
        """找到包含起始点的完整房间区域"""
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
            
            # 添加4连通的邻居
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return room_mask
    
    # 获取包含厨房中心的完整房间
    room_mask = flood_fill_room(center_x, center_y)
    room_pixels = np.sum(room_mask)
    
    if room_pixels < 100:  # 如果房间太小，不适合做厨房
        print(f"      ❌ 房间太小({room_pixels}像素)，不适合做厨房")
        return np.zeros((h, w), dtype=bool)
    
    print(f"      📏 发现房间区域: {room_pixels} 像素")
    
    # 计算房间的边界框
    room_coords = np.where(room_mask)
    min_y, max_y = np.min(room_coords[0]), np.max(room_coords[0])
    min_x, max_x = np.min(room_coords[1]), np.max(room_coords[1])
    room_width = max_x - min_x + 1
    room_height = max_y - min_y + 1
    
    print(f"      📐 房间边界: ({min_x},{min_y}) 到 ({max_x},{max_y}), 尺寸{room_width}x{room_height}")
    
    # 根据房间大小确定厨房尺寸（不能超过房间的80%）
    max_kitchen_width = int(room_width * 0.8)
    max_kitchen_height = int(room_height * 0.8)
    
    # 计算理想的厨房尺寸
    total_area = h * w
    target_area = min(total_area * 0.06, room_pixels * 0.7)  # 厨房最多占总面积6%或房间70%
    target_size = int(np.sqrt(target_area))
    
    # 限制厨房大小
    min_size = 20
    target_size = max(min_size, min(target_size, min(max_kitchen_width, max_kitchen_height)))
    
    print(f"      � 目标厨房尺寸: {target_size}x{target_size}")
    
    # 在房间内创建以中心点为中心的厨房区域
    half_size = target_size // 2
    
    # 确保厨房区域在房间边界内
    kitchen_left = max(min_x, center_x - half_size)
    kitchen_right = min(max_x + 1, center_x + half_size)
    kitchen_top = max(min_y, center_y - half_size)
    kitchen_bottom = min(max_y + 1, center_y + half_size)
    
    # 调整为正方形（在房间边界内）
    kitchen_width = kitchen_right - kitchen_left
    kitchen_height = kitchen_bottom - kitchen_top
    
    if kitchen_width < kitchen_height:
        # 尝试扩展宽度
        needed = kitchen_height - kitchen_width
        if kitchen_left - needed//2 >= min_x:
            kitchen_left -= needed//2
        elif kitchen_right + needed//2 <= max_x + 1:
            kitchen_right += needed//2
    elif kitchen_height < kitchen_width:
        # 尝试扩展高度
        needed = kitchen_width - kitchen_height
        if kitchen_top - needed//2 >= min_y:
            kitchen_top -= needed//2
        elif kitchen_bottom + needed//2 <= max_y + 1:
            kitchen_bottom += needed//2
    
    # 创建厨房掩码，只在房间区域内
    kitchen_mask = np.zeros((h, w), dtype=bool)
    
    for y in range(kitchen_top, kitchen_bottom):
        for x in range(kitchen_left, kitchen_right):
            if room_mask[y, x]:  # 只在房间区域内
                kitchen_mask[y, x] = True
    
    actual_width = kitchen_right - kitchen_left
    actual_height = kitchen_bottom - kitchen_top
    actual_pixels = np.sum(kitchen_mask)
    
    print(f"      ✅ 厨房区域生成完成:")
    print(f"         边界: ({kitchen_left},{kitchen_top}) 到 ({kitchen_right},{kitchen_bottom})")
    print(f"         尺寸: {actual_width}x{actual_height}")
    print(f"         有效像素: {actual_pixels}")
    
    return kitchen_mask

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
                
                # 获取厨房位置用于可视化标记
                kitchen_boxes = []
                if ocr_results:
                    for ocr_item in ocr_results:
                        text = ocr_item['text'].lower()
                        if any(keyword in text for keyword in ['厨房', 'kitchen', 'cook', '烹饪']):
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
                        cv2.putText(floorplan_original_size, ocr_text, 
                                  (ocr_x+35, ocr_y-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 添加白色背景
                        text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(floorplan_original_size,
                                    (ocr_x+33, ocr_y-35),
                                    (ocr_x+37+text_size[0], ocr_y-15),
                                    (255, 255, 255), -1)
                        cv2.putText(floorplan_original_size, ocr_text, 
                                  (ocr_x+35, ocr_y-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        print(f"🎯 OCR识别的厨房文字中心: 绿色框({ocr_x}, {ocr_y})")
                        print(f"� 这是'厨房'两个字的精确中心位置")
                        
                        # 添加图例说明
                        legend_y = 30
                        cv2.putText(floorplan_original_size, "绿色=OCR识别厨房文字中心", 
                                  (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(floorplan_original_size, "这是'厨房'两字的精确位置", 
                                  (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 保存带标记的结果
                    marked_filename = image_path.replace('.jpg', '_marked.png').replace('.png', '_marked.png')
                    imsave(marked_filename, floorplan_original_size)
                    print(f"✅ 带厨房标记的结果已保存: {marked_filename}")
                else:
                    # 没有厨房时，也添加坐标网格便于分析
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
                    plt.legend()
                
                # Save result
                output_name = os.path.basename(image_path).split('.')[0] + '_result.png'
                plt.savefig(output_name, dpi=300, bbox_inches='tight')
                print(f"📸 结果已保存: {output_name}")
                
                plt.show()

if __name__ == '__main__':
        FLAGS, unparsed = parser.parse_known_args()
        main(FLAGS)
