import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib
import cv2
from scipy import ndimage

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

parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')
parser.add_argument('--disable_closet', action='store_true',
                    help='map closet predictions to background')


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


def expand_kitchen_region_from_center(floorplan, center_x, center_y, original_shape):
    """从厨房中心点向四周墙壁边界延伸，画出整个厨房区域"""
    print(f"🏠 开始厨房区域扩展: 中心({center_x}, {center_y})")
    
    # 获取图像尺寸
    h, w = original_shape[:2]
    
    # 创建厨房掩码
    kitchen_mask = np.zeros((h, w), dtype=bool)
    
    # 使用区域增长算法从中心点扩展
    # 1. 首先标记中心点
    if 0 <= center_y < h and 0 <= center_x < w:
        kitchen_mask[center_y, center_x] = True
    
    # 2. 向四个方向扩展直到遇到墙壁（黑色像素或边界）
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
    
    for dx, dy in directions:
        # 从中心向每个方向扩展
        current_x, current_y = center_x, center_y
        
        while True:
            current_x += dx
            current_y += dy
            
            # 检查边界
            if current_x < 0 or current_x >= w or current_y < 0 or current_y >= h:
                break
            
            # 检查是否遇到墙壁（假设墙壁是标签9或10）
            if floorplan[current_y, current_x] in [9, 10]:  # 墙壁标签
                break
            
            # 标记为厨房区域
            kitchen_mask[current_y, current_x] = True
    
    # 3. 使用形态学操作填充小孔洞
    from scipy import ndimage
    kitchen_mask = ndimage.binary_fill_holes(kitchen_mask)
    
    # 4. 将扩展的区域标记为厨房
    expanded_pixels = np.sum(kitchen_mask)
    floorplan[kitchen_mask] = 7  # 厨房标签
    
    print(f"✅ 厨房区域扩展完成: 扩展了{expanded_pixels}个像素")
    
    return kitchen_mask


def enhance_kitchen_detection(floorplan, ocr_results):
    """Enhance kitchen detection using spatial analysis and OCR results.
    
    This function uses heuristics to better identify kitchen areas:
    1. OCR text detection for explicit kitchen labels (when available)
    2. Spatial analysis - kitchens are often smaller, rectangular rooms
    3. Simple connected component analysis (without scipy dependency)
    """
    enhanced = floorplan.copy()
    h, w = enhanced.shape
    
    # First, check for OCR-based kitchen detection
    kitchen_found_by_ocr = False
    if ocr_results:
        for ocr_item in ocr_results:
            text = ocr_item['text'].lower()
            if any(keyword in text for keyword in ['厨房', 'kitchen', 'cook', '烹饪']):
                kitchen_found_by_ocr = True
                print(f"🍳 OCR检测到厨房文字: '{ocr_item['text']}'")
                break
    
    if not kitchen_found_by_ocr:
        print("📍 OCR未检测到厨房文字，使用空间分析方法...")
    
    # Find regions labeled as living/dining (class 3)
    living_dining_mask = (enhanced == 3)
    
    if np.sum(living_dining_mask) == 0:
        print("❌ 未发现客厅/餐厅/厨房区域")
        return enhanced
    
    try:
        # Use simple connected component analysis
        labeled_regions, num_regions = simple_connected_components(living_dining_mask)
        
        print(f"🔍 发现 {num_regions} 个客厅/餐厅/厨房区域")
        
        region_stats = []
        
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_area = np.sum(region_mask)
            
            # Get bounding box of this region
            region_coords = np.where(region_mask)
            if len(region_coords[0]) == 0:
                continue
                
            min_y, max_y = np.min(region_coords[0]), np.max(region_coords[0])
            min_x, max_x = np.min(region_coords[1]), np.max(region_coords[1])
            region_height = max_y - min_y + 1
            region_width = max_x - min_x + 1
            
            # Calculate various metrics
            aspect_ratio = max(region_width, region_height) / min(region_width, region_height)
            density = region_area / (region_width * region_height)
            relative_area = region_area / (h * w)
            
            # Check if any OCR results suggest this is a kitchen
            has_kitchen_text = False
            if ocr_results:
                for ocr_item in ocr_results:
                    if any(keyword in ocr_item['text'].lower() for keyword in ['厨房', 'kitchen', 'cook', '烹饪']):
                        ocr_x, ocr_y, ocr_w, ocr_h = ocr_item['bbox']
                        # Check if OCR text is within this region
                        if (min_x <= ocr_x + ocr_w/2 <= max_x and min_y <= ocr_y + ocr_h/2 <= max_y):
                            has_kitchen_text = True
                            break
            
            region_stats.append({
                'id': region_id,
                'mask': region_mask,
                'area': region_area,
                'relative_area': relative_area,
                'aspect_ratio': aspect_ratio,
                'density': density,
                'width': region_width,
                'height': region_height,
                'has_kitchen_text': has_kitchen_text,
                'bbox': (min_x, min_y, max_x, max_y)
            })
        
        # Sort regions by area (smallest first - kitchens are often smaller)
        region_stats.sort(key=lambda x: x['relative_area'])
        
        # Enhanced heuristics for kitchen detection
        kitchen_assigned = False
        for i, stats in enumerate(region_stats):
            print(f"   区域{stats['id']}: 面积={stats['relative_area']:.3f}, 长宽比={stats['aspect_ratio']:.2f}, 密度={stats['density']:.2f}")
            
            is_kitchen = False
            reasons = []
            
            # Rule 1: Explicit OCR detection
            if stats['has_kitchen_text']:
                is_kitchen = True
                reasons.append("OCR检测到厨房文字")
            
            # Rule 2: Small area + reasonable shape (only assign one kitchen)
            elif (not kitchen_assigned and 
                  stats['relative_area'] < 0.15 and  # Smaller than 15% of total area
                  1.0 < stats['aspect_ratio'] < 4.0 and  # Not too elongated
                  stats['density'] > 0.5):  # Good density
                is_kitchen = True
                reasons.append("小面积+合理形状")
            
            # Rule 3: Very small compact area (only assign one kitchen)
            elif (not kitchen_assigned and
                  stats['relative_area'] < 0.08 and  # Very small area
                  stats['aspect_ratio'] < 3.0 and
                  stats['density'] > 0.6):
                is_kitchen = True
                reasons.append("紧凑型厨房")
            
            # Rule 4: If multiple regions, the smallest reasonable one might be kitchen
            elif (not kitchen_assigned and
                  len(region_stats) > 1 and i == 0 and  # Smallest region
                  stats['relative_area'] < 0.2 and
                  stats['aspect_ratio'] < 5.0 and
                  stats['density'] > 0.4):
                is_kitchen = True
                reasons.append("多区域中最小的合理区域")
            
            if is_kitchen:
                enhanced[stats['mask']] = 7  # Kitchen class
                print(f"   ✅ 识别为厨房: {', '.join(reasons)}")
                kitchen_assigned = True
            else:
                print(f"   ❌ 保持为客厅/餐厅")
                
        if not kitchen_assigned:
            print("⚠️ 未能自动识别厨房，所有区域保持为客厅/餐厅")
                
    except Exception as e:
        print(f"⚠️ 空间分析出错: {e}")
        import traceback
        traceback.print_exc()
    
    return enhanced

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
        enable_closet = not args.disable_closet
        set_closet_enabled(enable_closet)

        # load input
        im = imread(args.im_path, mode='RGB')
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

        # create tensorflow session with CPU configuration
        config = tf.ConfigProto(
                device_count={'GPU': 0},  # Disable GPU
                allow_soft_placement=True,
                log_device_placement=False
        )
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
                
                # 🎯 应用精确厨房坐标转换 - 直接使用OCR检测的厨房文字位置
                floorplan, kitchen_boxes = apply_precise_kitchen_coordinates(floorplan, ocr_results, original_im.shape[:2])
                
                # 🏠 从厨房中心向四周扩展到墙壁边界
                if kitchen_boxes:
                    for kitchen_info in kitchen_boxes:
                        center_x, center_y = kitchen_info['center']
                        # 将512x512坐标转换为原始图像坐标进行区域扩展
                        orig_center_x = int(center_x * original_im.shape[1] / 512)
                        orig_center_y = int(center_y * original_im.shape[0] / 512)
                        
                        # 创建原始尺寸的floorplan用于区域扩展
                        original_h, original_w = original_im.shape[:2]
                        floorplan_full_size = cv2.resize(floorplan.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        
                        print(f"🏠 开始厨房区域扩展: 从({orig_center_x}, {orig_center_y})向四周墙壁延伸")
                        kitchen_mask = expand_kitchen_region_from_center(floorplan_full_size, orig_center_x, orig_center_y, original_im.shape)
                        
                        # 将扩展结果缩放回512x512用于后续处理
                        kitchen_mask_512 = cv2.resize(kitchen_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
                        floorplan[kitchen_mask_512 > 0] = 7  # 将扩展区域标记为厨房
                
                # Use OCR labels to refine room categories
                floorplan = fuse_ocr_and_segmentation(floorplan, ocr_results)
                # Enhance kitchen detection (now with precise coordinates)
                floorplan = enhance_kitchen_detection(floorplan, ocr_results)
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
                    marked_filename = FLAGS.im_path.replace('.jpg', '_marked.png').replace('.png', '_marked.png')
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
                output_name = os.path.basename(args.im_path).split('.')[0] + '_result.png'
                plt.savefig(output_name, dpi=300, bbox_inches='tight')
                print(f"📸 结果已保存: {output_name}")
                
                plt.show()

if __name__ == '__main__':
        FLAGS, unparsed = parser.parse_known_args()
        main(FLAGS)
