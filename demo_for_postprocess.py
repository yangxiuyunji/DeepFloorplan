import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib

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

# Force CPU usage - disable GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='./demo/demo.jpg',
                    help='input image paths.')
parser.add_argument('--output_dir', type=str, default='./out',
                    help='output directory for predictions.')
parser.add_argument('--disable_closet', action='store_true',
                    help='map closet predictions to background')

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
    
    # Find regions labeled as living/dining (class 3)
    living_dining_mask = (enhanced == 3)
    
    if np.sum(living_dining_mask) == 0:
        print("❌ 未发现客厅/餐厅/厨房区域")
        return enhanced
    
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load input
    im = imread(args.im_path, mode='RGB')
    # Keep original size for better OCR
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
        
        # Use OCR labels to refine room categories
        floorplan = fuse_ocr_and_segmentation(floorplan, ocr_results)
        # Enhance kitchen detection
        floorplan = enhance_kitchen_detection(floorplan, ocr_results)
        
        if not enable_closet:
            floorplan[floorplan==1] = 0
        
        floorplan_rgb = ind2rgb(floorplan, enable_closet)

        # Save prediction result for postprocessing
        basename = os.path.basename(args.im_path).split('.')[0]
        pred_output_path = os.path.join(args.output_dir, f"{basename}_prediction.png")
        imsave(pred_output_path, (floorplan_rgb).astype(np.uint8))
        
        print(f"📸 预测结果已保存: {pred_output_path}")
        print(f"📁 准备后处理，输出目录: {args.output_dir}")

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
