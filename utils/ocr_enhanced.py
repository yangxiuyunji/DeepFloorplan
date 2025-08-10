"""Enhanced OCR utilities with PaddleOCR support for floorplan processing."""

from typing import List, Dict, Tuple, Any
import numpy as np

# Try PaddleOCR first
try:
    from paddleocr import PaddleOCR
    import cv2
    _HAS_PADDLE_OCR = True
    _paddle_ocr_instance = None
    print("🚀 PaddleOCR可用，将优先使用")
except Exception as e:
    _HAS_PADDLE_OCR = False
    print(f"❌ PaddleOCR导入失败: {e}")
    print("⚠️ 本程序需要PaddleOCR才能正常工作，请安装PaddleOCR")

# Fallback to Tesseract
try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image, ImageEnhance
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    Output = None
    _HAS_TESSERACT = False

# Text to label mapping (enhanced for all room types)
TEXT_LABEL_MAP = {
    # bedroom - 卧室类 (标签4)
    '卧室': 4, '主卧': 4, '次卧': 4, '卧室A': 4, '卧室B': 4, '卧室C': 4, '卧室1': 4, '卧室2': 4, '卧室3': 4,
    'bedroom': 4, 'br': 4, 'bed': 4, 'master': 4, 'guest': 4,
    
    # bathroom - 卫生间类 (标签2)  
    '卫生间': 2, '洗手间': 2, '浴室': 2, '卫A': 2, '卫B': 2, '卫1': 2, '卫2': 2,
    'bathroom': 2, 'washroom': 2, 'toilet': 2, 'wc': 2, 'bath': 2,
    
    # living/dining - 客厅餐厅类 (标签3)
    '客厅': 3, '餐厅': 3, '起居室': 3, '饭厅': 3, '会客厅': 3,
    'living': 3, 'livingroom': 3, 'living room': 3, 'dining': 3, 'diningroom': 3, 'dining room': 3,
    
    # balcony - 阳台类 (标签6)
    '阳台': 6, '露台': 6, '花园': 6, '庭院': 6, '平台': 6,
    'balcony': 6, 'terrace': 6, 'patio': 6, 'garden': 6, 'deck': 6,
    
    # kitchen - 厨房类 (标签7)
    '厨房': 7, '厨': 7, '烹饪间': 7, '料理台': 7,
    'kitchen': 7, 'cook': 7, '烹饪': 7, 'kitchenette': 7,
    
    # hall/entrance - 玄关大厅类 (标签5)
    '玄关': 5, '大厅': 5, '门厅': 5, '过道': 5, '走廊': 5, '入口': 5,
    'hall': 5, 'lobby': 5, 'entrance': 5, 'corridor': 5, 'foyer': 5,
    
    # closet/storage - 储物间类 (标签1)
    '衣柜': 1, '储物间': 1, '杂物间': 1, '储藏室': 1, '衣帽间': 1,
    'closet': 1, 'storage': 1, 'wardrobe': 1, 'pantry': 1
}

# Global closet control - disabled by default
ENABLE_CLOSET = False

# Room type descriptions for logging
ROOM_TYPE_NAMES = {
    1: "储物间",
    2: "卫生间", 
    3: "客厅/餐厅",
    4: "卧室",
    5: "玄关/大厅",
    6: "阳台",
    7: "厨房"
}

# Room type emojis for better visualization
ROOM_TYPE_EMOJIS = {
    1: "🗄️",
    2: "🚿", 
    3: "🛋️",
    4: "🛏️",
    5: "🚪",
    6: "🌿",
    7: "🍳"
}

def set_closet_enabled(enable: bool) -> None:
    """Globally enable or disable the closet category."""
    global ENABLE_CLOSET
    ENABLE_CLOSET = enable

def reset_paddle_ocr() -> None:
    """Reset PaddleOCR instance to apply new parameters"""
    global _paddle_ocr_instance
    _paddle_ocr_instance = None
    print("🔄 重置PaddleOCR实例")

def extract_room_text(image: Any) -> List[Dict]:
    """Extract room text using PaddleOCR only.
    
    Parameters
    ----------
    image: Any
        Input image (numpy array, PIL Image, or file path)
        
    Returns
    -------
    List[Dict]
        List of detected text with bounding boxes and confidence
    """
    if not _HAS_PADDLE_OCR:
        raise RuntimeError(
            "❌ PaddleOCR不可用！请确保已正确安装PaddleOCR：\n"
            "   pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "   pip install paddleocr==2.6.1.3\n"
            "   或者使用CPU版本：pip install paddlepaddle==2.5.2"
        )
    
    print("🎯 正在使用 PaddleOCR 进行中文文字识别...")
    return extract_room_text_paddle(image)

def extract_room_text_paddle(image: Any) -> List[Dict]:
    """Extract room text using PaddleOCR with enhanced parameters"""
    global _paddle_ocr_instance
    
    # Check if PaddleOCR is available
    if not _HAS_PADDLE_OCR:
        raise RuntimeError(
            "❌ PaddleOCR不可用！请确保已正确安装PaddleOCR：\n"
            "   pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "   pip install paddleocr==2.6.1.3\n"
            "   或者使用CPU版本：pip install paddlepaddle==2.5.2"
        )
    
    if _paddle_ocr_instance is None:
        print("🚀 初始化PaddleOCR（增强模式）...")
        try:
            # 设置环境变量来减少警告信息
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # 基于PaddleOCR官方文档的最佳参数配置
            _paddle_ocr_instance = PaddleOCR(
                lang='ch',
                det_db_thresh=0.2,        # 像素分类阈值，越小检测越多文本（官方默认0.3）
                det_db_box_thresh=0.4,    # 文本框置信度阈值，减少漏检（官方默认0.6）
                det_db_unclip_ratio=2.5,  # 文本框扩张系数，官方推荐2.5避免边缘丢失
                drop_score=0.3,           # 识别结果置信度过滤（默认0.5）
                use_angle_cls=True,       # 启用角度分类器处理旋转文字
                cls_thresh=0.8,           # 角度分类阈值（默认0.9）
                use_dilation=True,        # 官方推荐: 膨胀处理提高检测效果
                det_db_score_mode='slow'  # 官方推荐: 更精确的得分计算模式
            )
            print("✅ PaddleOCR专业优化模式初始化完成")
            print("   📋 官方最佳实践参数:")
            print("   🔹 det_db_thresh=0.2 (更敏感的像素检测)")
            print("   🔹 det_db_box_thresh=0.4 (减少漏检)")
            print("   🔹 det_db_unclip_ratio=2.5 (官方推荐扩张系数)")
            print("   🔹 use_dilation=True (膨胀处理提升效果)")
            print("   🔹 det_db_score_mode='slow' (精确得分计算)")
        except Exception as e:
            print(f"❌ PaddleOCR初始化失败: {e}")
            return []
    
    # Process image
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image)
    
    # Enhance image for better OCR and keep scale factor for coordinate correction
    enhanced_img, scale_factor = enhance_image_for_paddle_ocr(img)
    
    try:
        results = _paddle_ocr_instance.ocr(enhanced_img)
        extracted_texts = []
        
        # 处理新的PaddleOCR结果格式
        if results and len(results) > 0:
            result = results[0]  # 获取第一个结果
            
            # 尝试直接访问结果数据
            if 'rec_texts' in result:
                texts = result['rec_texts']
                scores = result['rec_scores']  
                polys = result['rec_polys']
                
                print(f"🔍 PaddleOCR检测到文本: {texts}")
                
                for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                    if score > 0.3 and text.strip():
                        # 从边界框多边形计算矩形边界
                        x_coords = [point[0] for point in poly]
                        y_coords = [point[1] for point in poly]
                        
                        # 坐标基于增强后的图像，需要缩放回原始尺寸
                        x = int(min(x_coords) / scale_factor)
                        y = int(min(y_coords) / scale_factor)
                        w = int((max(x_coords) - min(x_coords)) / scale_factor)
                        h = int((max(y_coords) - min(y_coords)) / scale_factor)
                        
                        extracted_texts.append({
                            'text': text,
                            'bbox': (x, y, w, h),
                            'confidence': score
                        })
                        print(f"🔍 PaddleOCR: '{text}' (置信度: {score:.3f})")
            
            # 兼容旧格式（如果有的话）
            elif isinstance(result, list):
                for line in result:
                    if len(line) >= 2:
                        bbox_points = line[0]
                        text_info = line[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                        else:
                            text = str(text_info)
                            confidence = 1.0
                        
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        
                        # 将坐标从增强图像缩放回原始图像
                        x = int(min(x_coords) / scale_factor)
                        y = int(min(y_coords) / scale_factor)
                        w = int((max(x_coords) - min(x_coords)) / scale_factor)
                        h = int((max(y_coords) - min(y_coords)) / scale_factor)
                        
                        if confidence > 0.3 and text.strip():
                            extracted_texts.append({
                                'text': text,
                                'bbox': (x, y, w, h),
                                'confidence': confidence
                            })
                            print(f"🔍 PaddleOCR: '{text}' (置信度: {confidence:.3f})")
        
        print(f"📊 PaddleOCR检测到 {len(extracted_texts)} 个文字区域")
        return extracted_texts
        
    except Exception as e:
        print(f"❌ PaddleOCR识别出错: {e}")
        print("📋 调试信息:")
        try:
            debug_results = _paddle_ocr_instance.ocr(enhanced_img)
            print(f"  原始结果类型: {type(debug_results)}")
            if debug_results:
                print(f"  结果数量: {len(debug_results)}")
                print(f"  完整结果结构: {debug_results}")
                
                # 安全地检查第一组结果
                if len(debug_results) > 0 and debug_results[0] is not None:
                    first_group = debug_results[0]
                    print(f"  第一组结果类型: {type(first_group)}")
                    print(f"  第一组结果数量: {len(first_group) if hasattr(first_group, '__len__') else 'N/A'}")
                    
                    if hasattr(first_group, '__len__') and len(first_group) > 0:
                        first_detection = first_group[0]
                        print(f"  第一个检测结果: {first_detection}")
                else:
                    print("  第一组结果为空或None")
            else:
                print("  结果为None或空")
        except Exception as debug_e:
            print(f"  调试失败: {debug_e}")
            import traceback
            print(f"  详细错误: {traceback.format_exc()}")
        return []

def enhance_image_for_paddle_ocr(image):
    """Optimize image for PaddleOCR.

    Returns
    -------
    Tuple[np.ndarray, float]
        Enhanced image and the scale factor applied.
    """
    height, width = image.shape[:2]
    scale_factor = max(2.0, 1000.0 / max(height, width))
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    enhanced = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    if len(enhanced.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=20)
    
    # Denoise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced, scale_factor

def extract_room_text_tesseract(image: Any) -> List[Dict]:
    """Extract room text using Tesseract (fallback)"""
    if not _HAS_TESSERACT:
        return []
    
    try:
        # Convert image to PIL format
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Enhance image
        pil_image = pil_image.convert('RGB')
        enhanced = enhance_image_for_tesseract(pil_image)
        
        # OCR
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz厨房客厅卧室卫生间阳台'
        data = pytesseract.image_to_data(enhanced, lang='chi_sim+eng', config=custom_config, output_type=Output.DICT)
        
        extracted_texts = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 30:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                extracted_texts.append({
                    'text': text,
                    'bbox': (x, y, w, h),
                    'confidence': conf / 100.0
                })
                print(f"🔍 Tesseract: '{text}' (置信度: {conf})")
        
        print(f"📊 Tesseract检测到 {len(extracted_texts)} 个文字区域")
        return extracted_texts
        
    except Exception as e:
        print(f"❌ Tesseract识别出错: {e}")
        return []

def enhance_image_for_tesseract(pil_image):
    """Optimize image for Tesseract"""
    # Resize
    width, height = pil_image.size
    scale_factor = max(2.0, 800.0 / max(width, height))
    new_size = (int(width * scale_factor), int(height * scale_factor))
    enhanced = pil_image.resize(new_size, Image.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(2.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(2.0)
    
    return enhanced

def text_to_label(text: str) -> int:
    """Convert recognized text to room label.
    
    Returns -1 if text doesn't correspond to any known room type.
    When ENABLE_CLOSET is False, closet-related text maps to background (0).
    """
    # Clean text
    cleaned_text = ''.join(c for c in text if c.isalnum() or c in '厨房客厅卧室卫生间阳台')
    
    label = TEXT_LABEL_MAP.get(cleaned_text.lower(), -1)
    
    # Fuzzy matching
    if label == -1:
        for key, value in TEXT_LABEL_MAP.items():
            if key in cleaned_text or cleaned_text in key:
                label = value
                break
    
    # Handle closet disable
    if label == 1 and not ENABLE_CLOSET:
        return 0
        
    return label

def fuse_ocr_and_segmentation(seg, ocr_results):
    """Fuse OCR results with segmentation map.
    
    Parameters
    ----------
    seg: np.ndarray
        2-D segmentation labels array
    ocr_results: List[Dict]
        OCR results from extract_room_text
        
    Returns
    -------
    np.ndarray
        Fused segmentation result
    """
    fused = seg.copy()
    
    print(f"🔗 融合 {len(ocr_results)} 个OCR结果")
    
    for ocr_item in ocr_results:
        text = ocr_item['text']
        confidence = ocr_item.get('confidence', 1.0)
        x, y, w, h = ocr_item['bbox']
        
        label = text_to_label(text)
        
        if label != -1:
            print(f"📝 应用OCR标签: '{text}' -> 标签{label} (置信度: {confidence:.3f})")
            
            # Ensure bbox is within image bounds
            x = max(0, min(x, seg.shape[1] - 1))
            y = max(0, min(y, seg.shape[0] - 1))
            x1 = max(0, min(x + w, seg.shape[1]))
            y1 = max(0, min(y + h, seg.shape[0]))
            
            if x1 > x and y1 > y:
                region = fused[y:y1, x:x1]
                if region.size > 0:
                    # Preserve boundaries (doors/windows and walls)
                    mask = np.isin(region, [9, 10])
                    region[~mask] = label
                    fused[y:y1, x:x1] = region
    
    # Handle closet disable globally
    if not ENABLE_CLOSET:
        fused[fused == 1] = 0
        
    return fused

# Export main functions
__all__ = ['extract_room_text', 'fuse_ocr_and_segmentation', 'text_to_label',
           'TEXT_LABEL_MAP', 'set_closet_enabled', 'ENABLE_CLOSET']
