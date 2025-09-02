#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
九宫格风水方位可视化工具
在原图上叠加九宫格，显示各方位及对应的八宅八星
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont
import io

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
    # 尝试常见的中文字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
    ]
    
    for font_path in font_paths:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # 如果找不到字体，使用默认字体
    try:
        return ImageFont.load_default()
    except:
        return None

def draw_text_with_background(draw, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=3):
    """在PIL图像上绘制带背景的文字"""
    x, y = position
    
    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 绘制背景矩形
    bg_x1 = x - padding
    bg_y1 = y - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + text_height + padding
    
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
    
    # 绘制文字
    draw.text((x, y), text, font=font, fill=text_color)
    
    return text_width, text_height

def get_luoshu_directions():
    """返回九宫格方位映射"""
    return {
        "东南": (2, 0), "南": (1, 0), "西南": (0, 0),
        "东": (2, 1),   "中": (1, 1), "西": (0, 1),
        "东北": (2, 2), "北": (1, 2), "西北": (0, 2)
    }
    """返回九宫格方位映射"""
    return {
        "东南": (2, 0), "南": (1, 0), "西南": (0, 0),
        "东": (2, 1),   "中": (1, 1), "西": (0, 1),
        "东北": (2, 2), "北": (1, 2), "西北": (0, 2)
    }

def get_bazhai_stars():
    """返回八宅八星对应关系 (坐北朝南)"""
    return {
        "东南": "生气", "南": "延年", "西南": "绝命",
        "东": "天医",   "中": "中宫", "西": "五鬼", 
        "东北": "六煞", "北": "伏位", "西北": "祸害"
    }

def get_star_colors():
    """返回八星对应的颜色"""
    return {
        "生气": (0, 255, 0),    # 绿色 - 最吉
        "延年": (0, 200, 0),    # 深绿 - 吉
        "天医": (0, 150, 255),  # 橙色 - 吉 
        "伏位": (255, 255, 0),  # 黄色 - 小吉
        "中宫": (128, 128, 128), # 灰色 - 中性
        "绝命": (0, 0, 255),    # 红色 - 大凶
        "五鬼": (0, 0, 200),    # 深红 - 凶
        "六煞": (0, 100, 255),  # 橙红 - 小凶
        "祸害": (0, 50, 200)    # 深橙红 - 凶
    }

def draw_luoshu_grid(image, overlay_alpha=0.3):
    """在图像上绘制九宫格和方位标注"""
    h, w = image.shape[:2]
    
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(image)
    pil_overlay = pil_image.copy()
    
    # 创建绘制对象
    draw = ImageDraw.Draw(pil_overlay)
    
    # 九宫格尺寸
    grid_w = w // 3
    grid_h = h // 3
    
    directions = get_luoshu_directions()
    stars = get_bazhai_stars()
    colors = get_star_colors()
    
    # 获取字体
    font_size = min(w, h) // 20
    font = get_chinese_font(font_size)
    small_font = get_chinese_font(font_size - 4)
    
    # 绘制九宫格
    for direction, (col, row) in directions.items():
        x1 = col * grid_w
        y1 = row * grid_h
        x2 = x1 + grid_w
        y2 = y1 + grid_h
        
        # 获取对应的星位和颜色
        star = stars.get(direction, "未知")
        color = colors.get(star, (128, 128, 128))
        
        # 绘制背景色块 (半透明效果通过后续合成实现)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255), width=2)
        
        # 添加方位文字
        center_x = x1 + grid_w // 2
        center_y = y1 + grid_h // 2
        
        # 方位名称
        direction_text = direction
        star_text = f"{star}星"
        
        # 绘制方位文字 (上方)
        if font:
            # 计算文字位置使其居中
            bbox = draw.textbbox((0, 0), direction_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = center_x - text_w // 2
            text_y = center_y - text_h - 8
            
            draw_text_with_background(draw, direction_text, (text_x, text_y), 
                                    font, (255, 255, 255), (0, 0, 0), padding=3)
        
        # 绘制星位文字 (下方)
        if small_font:
            bbox = draw.textbbox((0, 0), star_text, font=small_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = center_x - text_w // 2
            text_y = center_y + 8
            
            draw_text_with_background(draw, star_text, (text_x, text_y), 
                                    small_font, (255, 255, 255), (0, 0, 0), padding=2)
    
    # 转换回OpenCV格式
    overlay_cv2 = pil_to_cv2(pil_overlay)
    
    # 混合原图和overlay
    result = cv2.addWeighted(image, 1 - overlay_alpha, overlay_cv2, overlay_alpha, 0)
    return result

def draw_room_positions(image, rooms_data):
    """在图像上标注房间位置"""
    # 转换为PIL图像以支持中文
    pil_image = cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 获取字体
    font = get_chinese_font(16)
    
    for room in rooms_data:
        if 'center' in room:
            center_x = int(room['center']['x'])
            center_y = int(room['center']['y'])
            room_text = f"{room.get('type', 'Unknown')}{room.get('index', '')}"
            
            # 使用PIL绘制房间标签
            if font:
                draw_text_with_background(draw, room_text, 
                                        (center_x - 25, center_y - 30), 
                                        font, (255, 255, 255), (0, 0, 0), padding=3)
    
    # 转换回OpenCV格式并绘制圆点
    result = pil_to_cv2(pil_image)
    
    # 绘制房间中心点 (用OpenCV绘制几何图形)
    for room in rooms_data:
        if 'center' in room:
            center_x = int(room['center']['x'])
            center_y = int(room['center']['y'])
            
            # 绘制房间中心点
            cv2.circle(result, (center_x, center_y), 8, (255, 0, 255), -1)
            cv2.circle(result, (center_x, center_y), 10, (255, 255, 255), 2)
    
    return result

def add_legend(image):
    """添加图例说明"""
    h, w = image.shape[:2]
    legend_height = 150
    legend_width = w
    
    # 创建图例区域
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    legend[:] = (50, 50, 50)
    
    # 转换为PIL以绘制中文
    pil_legend = cv2_to_pil(legend)
    draw = ImageDraw.Draw(pil_legend)
    
    # 获取字体
    title_font = get_chinese_font(24)
    text_font = get_chinese_font(16)
    
    # 标题
    title = "八宅八星图例 (坐北朝南)"
    if title_font:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = bbox[2] - bbox[0]
        draw.text((w//2 - title_w//2, 10), title, font=title_font, fill=(255, 255, 255))
    
    # 星位说明
    stars_info = [
        ("生气星", "最吉", (0, 255, 0)),
        ("延年星", "吉", (0, 200, 0)),
        ("天医星", "吉", (0, 150, 255)),
        ("伏位星", "小吉", (255, 255, 0)),
        ("绝命星", "大凶", (0, 0, 255)),
        ("五鬼星", "凶", (0, 0, 200)),
        ("六煞星", "小凶", (0, 100, 255)),
        ("祸害星", "凶", (0, 50, 200))
    ]
    
    # 转换回OpenCV绘制颜色块
    legend_cv2 = pil_to_cv2(pil_legend)
    
    start_x = 50
    start_y = 60
    col_width = w // 4
    
    for i, (star, desc, color) in enumerate(stars_info):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 绘制颜色块
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), color, -1)
        cv2.rectangle(legend_cv2, (x, y), (x + 20, y + 20), (255, 255, 255), 1)
    
    # 再次转换为PIL绘制文字
    pil_legend = cv2_to_pil(legend_cv2)
    draw = ImageDraw.Draw(pil_legend)
    
    for i, (star, desc, color) in enumerate(stars_info):
        x = start_x + (i % 4) * col_width
        y = start_y + (i // 4) * 40
        
        # 文字
        text = f"{star} ({desc})"
        if text_font:
            draw.text((x + 30, y + 2), text, font=text_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    final_legend = pil_to_cv2(pil_legend)
    
    # 合并图例和原图
    result = np.vstack([image, final_legend])
    return result

def visualize_luoshu_grid(json_path, output_path=None):
    """生成九宫格可视化图像"""
    
    # 加载JSON数据
    data = load_json_data(json_path)
    
    # 获取图像路径
    meta = data.get('meta', {})
    original_image_path = meta.get('original_image')
    result_image_path = meta.get('result_image') or meta.get('output_image')
    
    # 确定使用的图像路径
    image_path = None
    json_dir = Path(json_path).parent
    project_root = json_dir.parent  # 假设JSON在output目录中
    
    # 优先使用原图
    if original_image_path:
        # 处理相对路径
        if '\\' in original_image_path or '/' in original_image_path:
            # 相对于项目根目录的路径
            image_path = project_root / original_image_path.replace('\\', '/')
        else:
            image_path = json_dir / original_image_path
            
    elif result_image_path:
        # 处理相对路径
        if '\\' in result_image_path or '/' in result_image_path:
            image_path = project_root / result_image_path.replace('\\', '/')
        else:
            image_path = json_dir / result_image_path
    
    if not image_path or not image_path.exists():
        # 尝试根据JSON文件名推测图像路径
        json_stem = Path(json_path).stem
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = json_dir / f"{json_stem.replace('_result', '')}{ext}"
            if candidate.exists():
                image_path = candidate
                break
    
    if not image_path or not image_path.exists():
        raise FileNotFoundError(f"找不到对应的图像文件，JSON路径: {json_path}")
    
    print(f"使用图像文件: {image_path}")
    
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像文件: {image_path}")
    
    # 绘制九宫格
    result_image = draw_luoshu_grid(image.copy())
    
    # 标注房间位置
    rooms = data.get('rooms', [])
    result_image = draw_room_positions(result_image, rooms)
    
    # 添加图例
    final_image = add_legend(result_image)
    
    # 保存结果
    if not output_path:
        output_path = json_dir / f"{Path(json_path).stem}_luoshu_grid.png"
    
    cv2.imwrite(str(output_path), final_image)
    print(f"九宫格可视化图已保存至: {output_path}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='生成九宫格风水方位可视化图')
    parser.add_argument('json_path', help='输入的JSON文件路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    
    args = parser.parse_args()
    
    try:
        output_path = visualize_luoshu_grid(args.json_path, args.output)
        print(f"✅ 九宫格可视化完成: {output_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
