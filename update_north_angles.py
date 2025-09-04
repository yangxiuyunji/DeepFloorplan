#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新JSON文件中的north_angle值从90改为0
"""

import json
import glob
from pathlib import Path

def update_north_angles():
    """更新所有JSON文件中的north_angle值"""
    json_files = glob.glob('./output/*_result_edited.json')
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            meta = data.get('meta', {})
            current_angle = meta.get('north_angle')
            
            if current_angle == 90:
                meta['north_angle'] = 0
                data['meta'] = meta
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f'✅ 已更新: {Path(file_path).name} (90° → 0°)')
            else:
                print(f'⏭️  跳过: {Path(file_path).name} (north_angle={current_angle})')
                
        except Exception as e:
            print(f'❌ 错误处理 {Path(file_path).name}: {e}')
    
    print(f"\n批量更新完成！")

if __name__ == "__main__":
    update_north_angles()
