#!/usr/bin/env python3
"""可视化房间分布和九宫格划分"""

import json
import cv2
import numpy as np
from pathlib import Path

def visualize_room_distribution(json_path):
    """可视化房间分布和九宫格划分"""
    
    # 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rooms = data.get("rooms", [])
    meta = data.get("meta", {})
    width = meta.get("image_width", 601)
    height = meta.get("image_height", 440)
    north_angle = meta.get("north_angle", 90)
    
    print(f"户型图尺寸: {width} x {height}")
    print(f"北向角度: {north_angle} (上方是{'南' if north_angle == 270 else '北'}方)")
    
    # 创建可视化图像
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 收集所有房间的边界框
    boxes = []
    for room in rooms:
        bbox = room.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            boxes.append((x1, y1, x2, y2))
            # 绘制房间框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 标注房间名称
            room_name = f"{room.get('type', '')} {room.get('index', '')}"
            cv2.putText(img, room_name, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 计算房屋的外接矩形
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    
    print(f"房屋边界: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    
    # 绘制房屋边界
    cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 3)
    
    # 九宫格划分
    grid_w = (max_x - min_x) / 3.0
    grid_h = (max_y - min_y) / 3.0
    
    print(f"九宫格尺寸: {grid_w:.1f} x {grid_h:.1f}")
    
    # 绘制九宫格线
    for i in range(1, 3):
        # 垂直线
        x = int(min_x + i * grid_w)
        cv2.line(img, (x, int(min_y)), (x, int(max_y)), (128, 128, 128), 2)
        # 水平线
        y = int(min_y + i * grid_h)
        cv2.line(img, (int(min_x), y), (int(max_x), y), (128, 128, 128), 2)
    
    # 九宫格方位映射（根据north_angle）
    if north_angle == 270:  # 上方是南方，右方是西方，左方是东方
        grid_directions = {
            (0, 0): "东南", (1, 0): "南", (2, 0): "西南",  # 上排：左东南，中南，右西南
            (0, 1): "东",   (1, 1): "中", (2, 1): "西",    # 中排：左东，中中，右西
            (0, 2): "东北", (1, 2): "北", (2, 2): "西北"   # 下排：左东北，中北，右西北
        }
    else:  # 默认上方是北方
        grid_directions = {
            (0, 0): "西北", (1, 0): "北", (2, 0): "东北",
            (0, 1): "西",   (1, 1): "中", (2, 1): "东",
            (0, 2): "西南", (1, 2): "南", (2, 2): "东南"
        }
    
    # 标注九宫格方位和覆盖率
    for gy in range(3):
        for gx in range(3):
            # 计算九宫格区域
            region_x1 = min_x + gx * grid_w
            region_x2 = min_x + (gx + 1) * grid_w
            region_y1 = min_y + gy * grid_h
            region_y2 = min_y + (gy + 1) * grid_h
            
            # 计算该区域被房间覆盖的面积
            region_area = (region_x2 - region_x1) * (region_y2 - region_y1)
            covered_area = 0
            
            for x1, y1, x2, y2 in boxes:
                # 计算房间与九宫格区域的重叠
                overlap_x1 = max(x1, region_x1)
                overlap_y1 = max(y1, region_y1)
                overlap_x2 = min(x2, region_x2)
                overlap_y2 = min(y2, region_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    covered_area += overlap_area
            
            # 计算覆盖率
            coverage_ratio = covered_area / region_area if region_area > 0 else 0
            
            # 获取方位名称
            direction = grid_directions.get((gx, gy), "未知")
            
            # 在九宫格中心标注方位和覆盖率
            center_x = int(region_x1 + grid_w / 2)
            center_y = int(region_y1 + grid_h / 2)
            
            # 根据覆盖率设置颜色
            color = (0, 255, 0) if coverage_ratio >= 0.6 else (0, 0, 255)
            
            # 绘制方位名称
            cv2.putText(img, direction, (center_x - 20, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 绘制覆盖率
            coverage_text = f"{coverage_ratio:.3f}"
            cv2.putText(img, coverage_text, (center_x - 25, center_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            print(f"{direction} ({gx},{gy}): 覆盖率 {coverage_ratio:.3f}")
    
    # 在图像上标注方向说明
    direction_text = f"北角度={north_angle}° (上方是{'南' if north_angle == 270 else '北'}方)"
    cv2.putText(img, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存图像
    output_path = "output/demo10_room_grid_analysis.png"
    cv2.imwrite(output_path, img)
    print(f"\n可视化图像已保存至: {output_path}")
    
    return output_path

if __name__ == "__main__":
    visualize_room_distribution("output/demo10_result_edited.json")
