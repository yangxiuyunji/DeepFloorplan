import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.ocr_enhanced import extract_room_text

# 测试坐标转换
def test_ocr_coordinates():
    # 加载图像
    image_path = './demo/demo.jpg'
    original_im = cv2.imread(image_path)
    original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
    
    print(f"原始图像尺寸: {original_im.shape}")
    
    # 创建OCR增强图像（放大2倍）
    ocr_img = Image.fromarray(original_im)
    ocr_img = ocr_img.resize((ocr_img.width * 2, ocr_img.height * 2), Image.LANCZOS)
    ocr_im = np.array(ocr_img)
    
    print(f"OCR图像尺寸: {ocr_im.shape}")
    
    # 提取OCR结果
    ocr_results = extract_room_text(ocr_im)
    
    # 创建结果图像
    result_img = original_im.copy()
    
    # 找到厨房并绘制
    for item in ocr_results:
        print(f"检测到文字: '{item['text']}' 位置: {item['bbox']}")
        if '厨房' in item['text'] or '廚房' in item['text']:  # 支持简体和繁体
            # OCR坐标（在2x图像中）
            ocr_x, ocr_y, ocr_w, ocr_h = item['bbox']
            print(f"OCR原始坐标: ({ocr_x}, {ocr_y}, {ocr_w}, {ocr_h})")
            
            # 转换到原始图像坐标
            orig_x = int(ocr_x * 0.5)  # 除以2
            orig_y = int(ocr_y * 0.5)
            orig_w = int(ocr_w * 0.5)
            orig_h = int(ocr_h * 0.5)
            
            print(f"转换到原始图像: ({orig_x}, {orig_y}, {orig_w}, {orig_h})")
            
            # 计算中心点
            center_x = orig_x + orig_w // 2
            center_y = orig_y + orig_h // 2
            print(f"中心点: ({center_x}, {center_y})")
            
            # 尝试不同的坐标修正方案
            # 方案1: 当前计算
            print(f"方案1 - 当前: 中心({center_x}, {center_y})")
            
            # 方案2: 使用左上角作为参考点（可能OCR检测框偏大）
            alt_center_x = orig_x + orig_w // 4  # 使用1/4位置而不是中心
            alt_center_y = orig_y + orig_h // 4
            print(f"方案2 - 1/4位置: 中心({alt_center_x}, {alt_center_y})")
            
            # 方案3: 考虑可能的系统性偏移  
            corrected_x = center_x - 132  # 更精确的X偏移修正
            corrected_y = center_y - 79   # Y偏移修正
            print(f"方案3 - 精确修正: 中心({corrected_x}, {corrected_y})")
            
            # 方案4: 基于OCR框的左上角位置计算（考虑文字在框内的相对位置）
            text_offset_x = orig_x + orig_w * 0.2  # 文字通常在框的左侧20%位置
            text_offset_y = orig_y + orig_h * 0.3  # 文字通常在框的上方30%位置
            print(f"方案4 - 文字偏移: 中心({int(text_offset_x)}, {int(text_offset_y)})")
            
            # 方案5: 多重分析方法
            final_x = int((corrected_x + text_offset_x) / 2)  # 平均两种方法
            final_y = int((corrected_y + text_offset_y) / 2)
            print(f"方案5 - 综合分析: 中心({final_x}, {final_y})")
            
            # 在原始图像上绘制多个方案的标记
            # 方案1 - 红色框（当前方法）
            cv2.rectangle(result_img, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (255, 0, 0), 3)
            cv2.circle(result_img, (center_x, center_y), 8, (255, 0, 0), -1)
            cv2.putText(result_img, f"1({center_x},{center_y})", (center_x+10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 方案2 - 绿色标记（1/4位置）
            cv2.circle(result_img, (alt_center_x, alt_center_y), 8, (0, 255, 0), -1)
            cv2.putText(result_img, f"2({alt_center_x},{alt_center_y})", (alt_center_x+10, alt_center_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 方案3 - 蓝色标记（精确偏移修正）
            if corrected_x > 0 and corrected_y > 0:
                cv2.circle(result_img, (corrected_x, corrected_y), 8, (0, 0, 255), -1)
                cv2.putText(result_img, f"3({corrected_x},{corrected_y})", (corrected_x+10, corrected_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 方案4 - 紫色标记（文字偏移）
            if text_offset_x > 0 and text_offset_y > 0:
                cv2.circle(result_img, (int(text_offset_x), int(text_offset_y)), 8, (128, 0, 128), -1)
                cv2.putText(result_img, f"4({int(text_offset_x)},{int(text_offset_y)})", (int(text_offset_x)+10, int(text_offset_y)-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)
            
            # 方案5 - 橙色标记（综合分析）
            if final_x > 0 and final_y > 0:
                cv2.circle(result_img, (final_x, final_y), 10, (255, 165, 0), -1)
                cv2.putText(result_img, f"5({final_x},{final_y})", (final_x+10, final_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            
            # 你说的目标位置 - 黄色标记
            target_x, target_y = 150, 100
            cv2.circle(result_img, (target_x, target_y), 12, (255, 255, 0), -1)
            cv2.putText(result_img, f"Target({target_x},{target_y})", (target_x+10, target_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            break
    
    # 显示结果
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.imshow(original_im)
    plt.title('原始图像')
    plt.axis('on')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    plt.imshow(result_img)
    plt.title('OCR厨房位置标记')
    plt.axis('on')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('coordinate_test.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_ocr_coordinates()
