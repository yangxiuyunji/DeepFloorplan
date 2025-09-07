#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证demo2户型图的north_angle值和二十四山旋转效果
"""

import json
from pathlib import Path

def check_demo2_north_angle():
    """检查demo2的north_angle值"""
    json_path = "output/demo2_result_edited.json"
    
    print(f"🔍 检查 {json_path} 的north_angle值...")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        north_angle = data.get('north_angle', 0)
        print(f"📊 当前north_angle: {north_angle}°")
        
        # 计算关键山位的旋转后位置
        print("\n🔄 关键山位旋转后位置:")
        key_mountains = {
            "子": 0,    # 正北
            "巽": 135,  # 东南
            "午": 180,  # 正南  
            "酉": 270   # 正西
        }
        
        for mountain, standard_angle in key_mountains.items():
            rotated_angle = (standard_angle + north_angle) % 360
            
            # 判断最终朝向
            if rotated_angle == 0:
                direction = "正上方（北）"
            elif rotated_angle == 90:
                direction = "正右方（东）"
            elif rotated_angle == 180:
                direction = "正下方（南）"
            elif rotated_angle == 270:
                direction = "正左方（西）"
            elif rotated_angle == 45:
                direction = "右上方（东北）"
            elif rotated_angle == 135:
                direction = "右下方（东南）"
            elif rotated_angle == 225:
                direction = "左下方（西南）"
            elif rotated_angle == 315:
                direction = "左上方（西北）"
            else:
                direction = f"{rotated_angle}°方向"
                
            print(f"  {mountain}: {standard_angle:3d}° + {north_angle}° = {rotated_angle:3d}° ({direction})")
        
        # 验证巽位是否在正上方
        xun_rotated = (135 + north_angle) % 360
        if xun_rotated == 0:
            print(f"\n✅ 巽位正确对准正上方！")
        else:
            print(f"\n⚠️ 巽位在{xun_rotated}°位置，不在正上方")
            
    except Exception as e:
        print(f"❌ 读取文件错误: {e}")

if __name__ == "__main__":
    check_demo2_north_angle()
