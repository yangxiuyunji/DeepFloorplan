#!/usr/bin/env python3
"""
可视化九宫格划分，检查西方区域
"""

import json
from editor.json_io import load_floorplan_json
from PIL import Image, ImageDraw, ImageFont

def visualize_grid_division():
    # 加载房间数据
    doc = load_floorplan_json("output/demo4_new_result_edited.json")
    
    # 创建图像
    img = Image.new('RGB', (doc.img_w, doc.img_h), 'white')
    draw = ImageDraw.Draw(img)
    
    # 收集房间边界框
    boxes = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, room_model in enumerate(doc.rooms):
        x1, y1, x2, y2 = room_model.bbox
        boxes.append((x1, y1, x2, y2))
        
        # 绘制房间边界
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 添加房间标签
        room_name = f"{room_model.type}{room_model.index}"
        draw.text((x1+2, y1+2), room_name, fill=color)
    
    # 计算外接矩形
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    
    # 绘制外接矩形
    draw.rectangle([min_x, min_y, max_x, max_y], outline='black', width=3)
    
    # 九宫格划分
    grid_w = (max_x - min_x) / 3.0
    grid_h = (max_y - min_y) / 3.0
    
    # 绘制九宫格线
    for i in range(1, 3):
        # 垂直线
        x = min_x + i * grid_w
        draw.line([(x, min_y), (x, max_y)], fill='blue', width=2)
        
        # 水平线
        y = min_y + i * grid_h
        draw.line([(min_x, y), (max_x, y)], fill='blue', width=2)
    
    # 九宫格方位映射
    grid_directions = {
        (0, 0): "西北",
        (1, 0): "北", 
        (2, 0): "东北",
        (0, 1): "西",
        (1, 1): "中",
        (2, 1): "东",
        (0, 2): "西南",
        (1, 2): "南",
        (2, 2): "东南"
    }
    
    # 标注九宫格区域
    for gy in range(3):
        for gx in range(3):
            region_x1 = min_x + gx * grid_w
            region_x2 = min_x + (gx + 1) * grid_w
            region_y1 = min_y + gy * grid_h
            region_y2 = min_y + (gy + 1) * grid_h
            
            direction = grid_directions.get((gx, gy), "未知")
            center_x = (region_x1 + region_x2) / 2
            center_y = (region_y1 + region_y2) / 2
            
            # 高亮西方区域
            if gx == 0 and gy == 1:  # 西方
                draw.rectangle([region_x1, region_y1, region_x2, region_y2], 
                             outline='red', width=4)
                # 添加半透明填充效果
                overlay = Image.new('RGBA', img.size, (255, 0, 0, 50))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([region_x1, region_y1, region_x2, region_y2], 
                                     fill=(255, 0, 0, 50))
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
            
            # 标注方位
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((center_x-10, center_y-10), direction, fill='black', font=font)
    
    # 保存图像
    img.save("output/grid_visualization.png")
    print("已保存九宫格可视化到: output/grid_visualization.png")
    
    # 输出西方区域的详细信息
    print(f"\n外接矩形: ({min_x}, {min_y}) ~ ({max_x}, {max_y})")
    print(f"九宫格尺寸: {grid_w:.1f} x {grid_h:.1f}")
    
    # 西方区域
    west_x1 = min_x
    west_x2 = min_x + grid_w
    west_y1 = min_y + grid_h
    west_y2 = min_y + 2 * grid_h
    
    print(f"\n西方区域: ({west_x1:.1f}, {west_y1:.1f}) ~ ({west_x2:.1f}, {west_y2:.1f})")
    print("该区域与以下房间重叠:")
    
    for room_model in doc.rooms:
        x1, y1, x2, y2 = room_model.bbox
        room_name = f"{room_model.type}{room_model.index}"
        
        # 计算重叠
        overlap_x1 = max(x1, west_x1)
        overlap_y1 = max(y1, west_y1)
        overlap_x2 = min(x2, west_x2)
        overlap_y2 = min(y2, west_y2)
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            room_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = overlap_area / room_area
            print(f"  {room_name}: 房间({x1}, {y1})~({x2}, {y2}), 重叠面积{overlap_area:.0f}, 占房间{overlap_ratio:.1%}")

if __name__ == "__main__":
    visualize_grid_division()
