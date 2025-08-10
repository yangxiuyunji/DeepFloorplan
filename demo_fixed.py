#!/usr/bin/env python3
"""
修改demo.py以直接保存RGB图像，避免matplotlib插值
"""
import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# 配置TensorFlow日志级别，完全静音冗长输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误
import warnings
warnings.filterwarnings('ignore')

tf.logging.set_verbosity(tf.logging.ERROR)  # 减少TensorFlow日志

# Configure Chinese font support for matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Disable TF 2.x behavior for compatibility
tf.disable_v2_behavior()

# OCR utilities - use enhanced version with PaddleOCR support
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

def imresize(img, size):
    """Resize image using PIL"""
    if len(img.shape) == 2:  # Grayscale
        img_pil = Image.fromarray(img, mode='L')
    else:  # RGB
        img_pil = Image.fromarray(img, mode='RGB')
    
    resized = img_pil.resize((size[1], size[0]), Image.LANCZOS)
    return np.array(resized)

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('im_path', type=str, nargs='?', default='./demo/demo.jpg',
                    help='input image path')
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

def enhance_kitchen_detection(floorplan, ocr_results):
    """Enhance kitchen detection using spatial analysis and OCR results."""
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
    
    try:
        # Find all areas marked as living_room/dining_room/kitchen (class 3)
        living_areas = (enhanced == 3)
        
        if not np.any(living_areas):
            print("⚠️ 未找到客厅/餐厅/厨房区域")
            return enhanced
        
        # Use simple connected components
        labels, num_regions = simple_connected_components(living_areas)
        
        if num_regions == 0:
            print("⚠️ 连通区域分析失败")
            return enhanced
        
        print(f"🔍 发现 {num_regions} 个客厅/餐厅/厨房区域")
        
        # Analyze each region
        region_stats = []
        for region_id in range(1, num_regions + 1):
            region_mask = (labels == region_id)
            region_coords = np.where(region_mask)
            
            if len(region_coords[0]) == 0:
                continue
                
            # Calculate region properties
            min_y, max_y = region_coords[0].min(), region_coords[0].max()
            min_x, max_x = region_coords[1].min(), region_coords[1].max()
            
            region_height = max_y - min_y + 1
            region_width = max_x - min_x + 1
            region_area = np.sum(region_mask)
            
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
    """Convert indexed image to RGB"""
    # Use the appropriate color map based on closet setting
    if enable_closet:
        color_map = floorplan_fuse_map_figure
    else:
        # Create a modified map without closet
        color_map = floorplan_fuse_map_figure.copy()
        color_map[1] = color_map[3]  # Map closet to living room color
    
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3), dtype=np.uint8)

    for i, rgb in color_map.items():
        rgb_im[(ind_im==i)] = rgb

    return rgb_im

def main(args):
    enable_closet = not args.disable_closet
    set_closet_enabled(enable_closet)

    # Load input
    im = imread(args.im_path, mode='RGB')
    original_im = im.copy()
    
    # Resize image for network inference
    im = imresize(im, (512, 512))
    
    # For OCR, use larger, enhanced image
    from PIL import Image, ImageEnhance
    ocr_img = Image.fromarray(original_im)
    # Enlarge for better OCR
    ocr_img = ocr_img.resize((ocr_img.width * 2, ocr_img.height * 2), Image.LANCZOS)
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(ocr_img)
    ocr_img = enhancer.enhance(2.5)
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(ocr_img)
    ocr_img = enhancer.enhance(2.0)
    ocr_im = np.array(ocr_img)
    
    # Extract textual room labels using OCR with enhanced image
    ocr_results = extract_room_text(ocr_im)
    # Scale OCR bounding boxes to match segmentation size (512x512)
    if ocr_results:
        scale_x = im.shape[1] / ocr_im.shape[1]
        scale_y = im.shape[0] / ocr_im.shape[0]
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
            
            # Initialize
            sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Restore pretrained model
            saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
            saver.restore(sess, './pretrained/pretrained_r3d')

            # Get default graph
            graph = tf.get_default_graph()

            # Restore inputs & outputs tensor
            x = graph.get_tensor_by_name('inputs:0')
            room_type_logit = graph.get_tensor_by_name('Cast:0')
            room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

            # Infer results
            [room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],
                                                            feed_dict={x:im.reshape(1,512,512,3)})
            room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

            # Merge results
            floorplan = room_type.copy()
            floorplan[room_boundary==1] = 9
            floorplan[room_boundary==2] = 10
            
            # Use OCR labels to refine room categories
            floorplan = fuse_ocr_and_segmentation(floorplan, ocr_results)
            
            # Enhance kitchen detection
            floorplan = enhance_kitchen_detection(floorplan, ocr_results)
            
            # If closet is disabled, map closet areas to bedroom (2)
            if not enable_closet:
                floorplan[floorplan==1] = 2  # Map closet to bedroom
                    
            # Convert to RGB
            floorplan_rgb = ind2rgb(floorplan, True)  # Always use full color map

            # Save raw RGB result directly using PIL
            output_name = os.path.basename(args.im_path).split('.')[0] + '_raw_result.png'
            result_img = Image.fromarray(floorplan_rgb, mode='RGB')
            result_img.save(output_name)
            print(f"📸 原始RGB结果已保存: {output_name}")
            
            # Also create matplotlib version for comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(original_im)
            plt.title('原始图片')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(floorplan_rgb)
            plt.title('户型分析结果 (绿色=厨房)')
            plt.axis('off')
            
            # Save matplotlib result
            matplotlib_output = os.path.basename(args.im_path).split('.')[0] + '_matplotlib_result.png'
            plt.savefig(matplotlib_output, dpi=300, bbox_inches='tight')
            print(f"📸 Matplotlib结果已保存: {matplotlib_output}")
            
            plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
