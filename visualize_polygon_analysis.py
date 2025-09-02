#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化多边形和九宫格分析覆盖率问题
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from debug_coverage_detail import create_polygon_from_rooms
from editor.json_io import load_floorplan_json

def visualize_polygon_grid():
    """可视化多边形和九宫格，分析覆盖率"""
    
    # 加载数据
    json_path = 'output/demo4_new_result_edited.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    doc = load_floorplan_json(json_path)
    rooms = raw.get('rooms', [])
    polygon = create_polygon_from_rooms(rooms)
    
    # 计算边界
    pts = np.array(polygon, dtype=np.float32)
    min_x = float(np.min(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_x = float(np.max(pts[:, 0]))
    max_y = float(np.max(pts[:, 1]))
    
    # 九宫格划分
    grid_w = (max_x - min_x) / 3.0
    grid_h = (max_y - min_y) / 3.0
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制多边形
    poly_patch = MplPolygon(polygon, alpha=0.3, facecolor='blue', edgecolor='blue', linewidth=2)
    ax.add_patch(poly_patch)
    
    # 绘制多边形顶点
    for i, (x, y) in enumerate(polygon):
        ax.plot(x, y, 'ro', markersize=8)
        ax.annotate(f'P{i}({x:.0f},{y:.0f})', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 绘制九宫格
    position_names = [
        ["西北", "北", "东北"],
        ["西", "中", "东"],
        ["西南", "南", "东南"]
    ]
    
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'brown']
    color_idx = 0
    
    for gy in range(3):
        for gx in range(3):
            x1 = min_x + gx * grid_w
            x2 = min_x + (gx + 1) * grid_w
            y1 = min_y + gy * grid_h
            y2 = min_y + (gy + 1) * grid_h
            
            # 绘制九宫格边框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor=colors[color_idx % len(colors)], linewidth=2)
            ax.add_patch(rect)
            
            # 标注方位名称
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.text(center_x, center_y, position_names[gy][gx], 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            color_idx += 1
    
    # 设置坐标轴
    ax.set_xlim(min_x - 20, max_x + 20)
    ax.set_ylim(min_y - 20, max_y + 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('多边形与九宫格覆盖分析', fontsize=14, fontweight='bold')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    
    # 反转Y轴（因为图像坐标系Y轴向下）
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('polygon_grid_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 详细分析东北角
    print('\\n=== 东北角详细分析 ===')
    gx, gy = 2, 0  # 东北角
    x1 = min_x + gx * grid_w
    x2 = min_x + (gx + 1) * grid_w
    y1 = min_y + gy * grid_h
    y2 = min_y + (gy + 1) * grid_h
    
    print(f'东北角区域: ({x1:.1f}, {y1:.1f}) ~ ({x2:.1f}, {y2:.1f})')
    print(f'区域尺寸: {x2-x1:.1f} x {y2-y1:.1f}')
    
    # 使用OpenCV计算精确覆盖率
    mask = np.zeros((doc.img_h, doc.img_w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    
    # 提取该区域
    region_x1 = max(0, min(int(x1), doc.img_w))
    region_x2 = max(0, min(int(x2), doc.img_w))
    region_y1 = max(0, min(int(y1), doc.img_h))
    region_y2 = max(0, min(int(y2), doc.img_h))
    
    cell = mask[region_y1:region_y2, region_x1:region_x2]
    total = cell.size
    covered = cell.sum()
    ratio = float(covered) / float(total) if total > 0 else 0.0
    
    print(f'实际计算区域: ({region_x1}, {region_y1}) ~ ({region_x2}, {region_y2})')
    print(f'总像素: {total}, 覆盖像素: {covered}, 覆盖率: {ratio:.3f}')
    
    # 分析多边形在该区域的形状
    print('\\n该区域相关的多边形边:')
    for i in range(len(polygon)):
        x1_poly, y1_poly = polygon[i]
        x2_poly, y2_poly = polygon[(i+1) % len(polygon)]
        
        # 检查这条边是否经过东北角区域
        if ((x1 <= x1_poly <= x2 or x1 <= x2_poly <= x2) and 
            (y1 <= y1_poly <= y2 or y1 <= y2_poly <= y2)):
            print(f'  边{i}-{(i+1)%len(polygon)}: ({x1_poly:.1f},{y1_poly:.1f}) -> ({x2_poly:.1f},{y2_poly:.1f})')

if __name__ == "__main__":
    visualize_polygon_grid()
