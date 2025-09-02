#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
九宫格方位可视化工具 - 只显示九宫格划分
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_json_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cv2_to_pil(cv2_image):
    """将OpenCV图像转换为PIL图像"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    """将PIL图像转换为OpenCV图像"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_chinese_font(size=20):
    """获取中文字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
    ]
    
    for font_path in font_paths:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
        except:
            continue
    
    try:
        return ImageFont.load_default()
    except:
        return None

def draw_text_with_background(draw, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=3):
    """在PIL图像上绘制带背景的文字"""
    x, y = position
    
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制背景矩形
        bg_x1 = x - padding
        bg_y1 = y - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + text_height + padding
        
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
        draw.text((x, y), text, font=font, fill=text_color)

def get_luoshu_directions():
    """返回九宫格方位映射"""
    return {
        "东南": (2, 0), "南": (1, 0), "西南": (0, 0),
        "东": (2, 1),   "中": (1, 1), "西": (0, 1),
        "东北": (2, 2), "北": (1, 2), "西北": (0, 2)
    }

def draw_luoshu_grid_only(image):
    """绘制纯九宫格"""
    h, w = image.shape[:2]
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 九宫格尺寸
    grid_w = w // 3
    grid_h = h // 3
    
    directions = get_luoshu_directions()
    
    # 获取字体
    font_size = min(w, h) // 15
    font = get_chinese_font(font_size)
    
    # 绘制九宫格
    for direction, (col, row) in directions.items():
        x1 = col * grid_w
        y1 = row * grid_h
        x2 = x1 + grid_w
        y2 = y1 + grid_h
        
        # 绘制边框
        draw.rectangle([x1, y1, x2, y2], fill=None, outline=(255, 255, 0), width=3)
        
        # 添加方位文字
        center_x = x1 + grid_w // 2
        center_y = y1 + grid_h // 2
        
        if font:
            bbox = draw.textbbox((0, 0), direction, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = center_x - text_w // 2
            text_y = center_y - text_h // 2
            
            draw_text_with_background(draw, direction, (text_x, text_y), 
                                    font, (255, 255, 255), (0, 0, 0), padding=5)
    
    # 转换回OpenCV格式
    result = pil_to_cv2(pil_image)
    return result

def draw_room_positions(image, rooms_data):
    """在图像上标注房间位置"""
    pil_image = cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    
    font = get_chinese_font(14)
    
    for room in rooms_data:
        if 'center' in room:
            center_x = int(room['center']['x'])
            center_y = int(room['center']['y'])
            room_text = f"{room.get('type', 'Unknown')}{room.get('index', '')}"
            
            if font:
                bbox = draw.textbbox((0, 0), room_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_x = center_x - text_w // 2
                text_y = center_y - 25
                
                draw_text_with_background(draw, room_text, (text_x, text_y), 
                                        font, (255, 255, 255), (255, 0, 0), padding=3)
    
    # 转换回OpenCV格式并绘制圆点
    result = pil_to_cv2(pil_image)
    
    for room in rooms_data:
        if 'center' in room:
            center_x = int(room['center']['x'])
            center_y = int(room['center']['y'])
            
            cv2.circle(result, (center_x, center_y), 6, (255, 0, 0), -1)
            cv2.circle(result, (center_x, center_y), 8, (255, 255, 255), 2)
    
    return result

def add_luoshu_legend(image):
    """添加九宫格图例"""
    h, w = image.shape[:2]
    legend_height = 80
    
    legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)
    
    pil_legend = cv2_to_pil(legend)
    draw = ImageDraw.Draw(pil_legend)
    
    title_font = get_chinese_font(20)
    text_font = get_chinese_font(14)
    
    if title_font:
        title = "九宫格方位划分图"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = bbox[2] - bbox[0]
        draw.text((w//2 - title_w//2, 10), title, font=title_font, fill=(255, 255, 255))
    
    if text_font:
        desc = "说明：黄色线条将户型分为九个方位区域，红点标示房间位置"
        bbox = draw.textbbox((0, 0), desc, font=text_font)
        desc_w = bbox[2] - bbox[0]
        draw.text((w//2 - desc_w//2, 45), desc, font=text_font, fill=(200, 200, 200))
    
    final_legend = pil_to_cv2(pil_legend)
    result = np.vstack([image, final_legend])
    return result

def visualize_luoshu_grid_only(json_path, output_path=None):
    """生成纯九宫格可视化图像"""
    # 加载JSON数据
    data = load_json_data(json_path)
    
    # 获取图像路径
    meta = data.get('meta', {})
    original_image_path = meta.get('original_image')
    result_image_path = meta.get('result_image') or meta.get('output_image')
    
    # 确定使用的图像路径
    image_path = None
    json_dir = Path(json_path).parent
    project_root = json_dir.parent
    
    if original_image_path:
        if '\\' in original_image_path or '/' in original_image_path:
            image_path = project_root / original_image_path.replace('\\', '/')
        else:
            image_path = json_dir / original_image_path
            
    elif result_image_path:
        if '\\' in result_image_path or '/' in result_image_path:
            image_path = project_root / result_image_path.replace('\\', '/')
        else:
            image_path = json_dir / result_image_path
    
    if not image_path or not image_path.exists():
        raise FileNotFoundError(f"找不到对应的图像文件，JSON路径: {json_path}")
    
    print(f"使用图像文件: {image_path}")
    
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像文件: {image_path}")
    
    # 绘制九宫格
    result_image = draw_luoshu_grid_only(image.copy())
    
    # 标注房间位置
    rooms = data.get('rooms', [])
    result_image = draw_room_positions(result_image, rooms)
    
    # 添加图例
    final_image = add_luoshu_legend(result_image)
    
    # 保存结果
    if not output_path:
        output_path = json_dir / f"{Path(json_path).stem}_luoshu_grid_only.png"
    
    cv2.imwrite(str(output_path), final_image)
    print(f"九宫格可视化图已保存至: {output_path}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='生成九宫格方位划分可视化图')
    parser.add_argument('json_path', help='输入的JSON文件路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    
    args = parser.parse_args()
    
    try:
        output_path = visualize_luoshu_grid_only(args.json_path, args.output)
        print(f"✅ 九宫格可视化完成: {output_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
