#!/usr/bin/env python3
"""
DeepFloorplan颜色编码可视化展示
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 颜色映射定义
floorplan_map = {
    0: [255,255,255], # background
    1: [192,192,224], # closet  
    2: [192,255,255], # bathroom/washroom
    3: [224,255,192], # livingroom/kitchen/dining room
    4: [255,224,128], # bedroom
    5: [255,160, 96], # hall
    6: [255,224,224], # balcony
    7: [255,255,255], # not used
    8: [255,255,255], # not used
    9: [255, 60,128], # door & window
    10:[  0,  0,  0]  # wall
}

# 标签名称
labels = {
    0: "Background (背景)",
    1: "Closet (衣柜)",  
    2: "Bathroom (卫生间)",
    3: "Living/Kitchen/Dining (客厅/厨房/餐厅)",
    4: "Bedroom (卧室)",
    5: "Hall (走廊)",
    6: "Balcony (阳台)",
    7: "Not used (未使用)",
    8: "Not used (未使用)", 
    9: "Door & Window (门窗)",
    10: "Wall (墙体)"
}

def show_color_legend():
    """显示颜色图例"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建颜色块
    y_pos = 0
    patches = []
    
    for idx in range(11):
        if idx in [7, 8]:  # 跳过未使用的类别
            continue
            
        color = np.array(floorplan_map[idx]) / 255.0  # 归一化到0-1
        label = labels[idx]
        
        # 创建颜色块
        rect = Rectangle((0, y_pos), 2, 0.8, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # 添加文本标签
        ax.text(2.2, y_pos + 0.4, f"{idx}: {label}", 
                va='center', ha='left', fontsize=12, fontweight='bold')
        
        # 添加RGB值
        rgb_text = f"RGB({floorplan_map[idx][0]}, {floorplan_map[idx][1]}, {floorplan_map[idx][2]})"
        ax.text(6.5, y_pos + 0.4, rgb_text, 
                va='center', ha='left', fontsize=10, color='gray')
        
        y_pos += 1
    
    # 设置图形属性
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, y_pos + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加标题
    plt.title('DeepFloorplan 颜色编码图例\nColor Legend for DeepFloorplan Results', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 添加说明文字
    plt.figtext(0.1, 0.05, 
                "使用说明: 运行demo后，识别结果中不同颜色的区域对应上述房间类型和结构元素",
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('color_legend.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sample_floorplan():
    """创建一个示例户型图展示"""
    # 创建一个简单的示例
    sample = np.zeros((100, 150), dtype=np.uint8)
    
    # 背景
    sample[:, :] = 0
    
    # 卧室
    sample[10:40, 10:60] = 4
    
    # 客厅
    sample[10:40, 70:140] = 3
    
    # 卫生间  
    sample[50:80, 10:40] = 2
    
    # 厨房/餐厅
    sample[50:80, 50:140] = 3
    
    # 走廊
    sample[40:50, :] = 5
    
    # 阳台
    sample[85:95, 100:140] = 6
    
    # 墙体
    sample[0:10, :] = 10  # 上墙
    sample[90:100, :] = 10  # 下墙
    sample[:, 0:10] = 10  # 左墙
    sample[:, 140:150] = 10  # 右墙
    sample[40:50, 60:70] = 10  # 内墙
    
    # 门
    sample[45:48, 30:35] = 9
    sample[45:48, 95:100] = 9
    
    # 转换为RGB
    rgb_sample = np.zeros((100, 150, 3), dtype=np.uint8)
    for i, color in floorplan_map.items():
        rgb_sample[sample == i] = color
    
    return rgb_sample

def show_sample_result():
    """显示示例识别结果"""
    sample_rgb = create_sample_floorplan()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 显示示例结果
    ax1.imshow(sample_rgb)
    ax1.set_title('示例识别结果\nSample Recognition Result', fontweight='bold')
    ax1.axis('off')
    
    # 显示颜色图例
    legend_elements = []
    for idx, (color, label) in zip(floorplan_map.keys(), labels.values()):
        if idx in [7, 8]:  # 跳过未使用的类别
            continue
        color_norm = np.array(floorplan_map[idx]) / 255.0
        legend_elements.append(mpatches.Patch(color=color_norm, label=f"{idx}: {label}"))
    
    ax2.legend(handles=legend_elements, loc='center', fontsize=10)
    ax2.set_title('颜色图例\nColor Legend', fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_result.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== DeepFloorplan 颜色编码展示 ===")
    print("生成颜色图例和示例结果...")
    
    # 显示颜色图例
    show_color_legend()
    
    # 显示示例结果
    show_sample_result()
    
    print("图例已保存为: color_legend.png")
    print("示例结果已保存为: sample_result.png")
    print("\n颜色编码说明:")
    for idx, label in labels.items():
        if idx in [7, 8]:
            continue
        rgb = floorplan_map[idx]
        print(f"  {idx}: {label} - RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
