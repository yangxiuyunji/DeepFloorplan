#!/usr/bin/env python3
"""
DeepFloorplan颜色编码说明 - 文本版本
"""

# 颜色映射与标签定义
#
# 默认情况下, 类别 1 (衣柜) 会单独着色并输出。如果下游任务不需要
# 识别衣柜, 可以将其映射到背景。为方便配置, 提供两个辅助函数
# ``get_floorplan_map`` 和 ``get_labels``。

floorplan_map = {
    0: [255,255,255], # background
    1: [192,192,224], # closet
    2: [192,255,255], # bathroom/washroom
    3: [224,255,192], # livingroom/dining room
    4: [255,224,128], # bedroom
    5: [255,160, 96], # hall
    6: [255,224,224], # balcony
    7: [255,200,100], # kitchen - 新增厨房类别
    8: [255,255,255], # not used
    9: [255, 60,128], # door & window
    10:[  0,  0,  0]  # wall
}

labels = {
    0: "Background (背景)",
    1: "Closet (衣柜)",
    2: "Bathroom (卫生间)",
    3: "Living/Dining (客厅/餐厅)",
    4: "Bedroom (卧室)",
    5: "Hall (走廊)",
    6: "Balcony (阳台)",
    7: "Kitchen (厨房)",
    8: "Not used (未使用)",
    9: "Door & Window (门窗)",
    10: "Wall (墙体)"
}


def get_floorplan_map(enable_closet=True):
    """Return a copy of the color map.

    Parameters
    ----------
    enable_closet: bool
        If ``False``, category 1 (closet) will reuse the background color.
    """
    cmap = floorplan_map.copy()
    if not enable_closet:
        cmap[1] = floorplan_map[0]
    return cmap


def get_labels(enable_closet=True):
    """Return a copy of the label dictionary.

    When ``enable_closet`` is ``False`` the closet label is mapped to
    background so that图例和可视化不会显示衣柜。"""
    lbl = labels.copy()
    if not enable_closet:
        lbl[1] = labels[0]
    return lbl

def print_color_legend(enable_closet=True):
    """打印颜色图例。

    参数
    ------
    enable_closet: bool
        控制是否在图例中显示衣柜类别。
    """
    print("=" * 80)
    print("🎨 DeepFloorplan 识别结果颜色编码说明")
    print("=" * 80)
    print()
    
    print("📋 房间类型和结构元素对应的颜色:")
    print("-" * 80)
    
    cmap = get_floorplan_map(enable_closet)
    lbls = get_labels(enable_closet)

    for idx in range(11):
        if idx in [7, 8]:  # 跳过未使用的类别
            continue

        rgb = cmap[idx]
        label = lbls[idx]
        
        # 创建颜色的近似文本表示
        color_desc = get_color_description(rgb)
        
        print(f"  {idx:2d}: {label:<35} | RGB({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d}) | {color_desc}")
    
    print("-" * 80)
    print()

def get_color_description(rgb):
    """根据RGB值返回颜色描述"""
    r, g, b = rgb
    
    if r == 255 and g == 255 and b == 255:
        return "⬜ 白色"
    elif r == 0 and g == 0 and b == 0:
        return "⬛ 黑色"
    elif r > 220 and g > 220 and b > 200:
        return "🟦 淡紫色"
    elif r < 200 and g > 240 and b > 240:
        return "🟦 浅青色" 
    elif r > 220 and g > 240 and r < 230:
        return "🟩 浅绿色"
    elif r > 240 and g > 200 and b < 150:
        return "🟨 浅橙色"
    elif r > 240 and g < 180 and b < 120:
        return "🟧 橙色"
    elif r > 240 and g > 200 and b > 200:
        return "🟥 浅粉色"
    elif r > 240 and g < 100 and b > 100:
        return "🟥 深粉色"
    else:
        return "🎨 其他颜色"

def print_usage_guide():
    """打印使用说明"""
    print("📖 使用说明:")
    print("-" * 40)
    print("1. 运行命令: python demo.py --im_path=./demo/45719584.jpg")
    print("2. 程序会显示两个图像窗口:")
    print("   - 左图: 原始户型图")
    print("   - 右图: AI识别结果 (按上述颜色编码)")
    print()
    print("🏠 识别功能:")
    print("   ✓ 自动识别不同房间类型 (卧室、客厅、卫生间等)")
    print("   ✓ 检测建筑结构元素 (墙体、门窗)")
    print("   ✓ 理解户型图的空间布局")
    print()

def print_technical_details():
    """打印技术细节"""
    print("🔧 技术细节:")
    print("-" * 40)
    print("• 模型架构: 多任务神经网络")
    print("• 输出分支:")
    print("  - 房间类型分支: 识别功能区域 (0-6)")
    print("  - 边界分支: 识别墙体和开口 (0-2)")
    print("• 后处理: 将两个分支结果融合为最终输出")
    print("• 融合规则:")
    print("  - 基础: 房间类型结果")
    print("  - 覆盖: 边界检测结果 (门窗=9, 墙体=10)")
    print()

if __name__ == "__main__":
    print_color_legend()
    print_usage_guide() 
    print_technical_details()
    
    print("=" * 80)
    print("🎯 现在你可以运行 demo 来看实际的识别效果了！")
    print("命令: python demo.py --im_path=./demo/45719584.jpg")
    print("=" * 80)
