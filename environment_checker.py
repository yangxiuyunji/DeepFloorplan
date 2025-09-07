#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI助手工作环境配置检查器
确保每次运行都激活虚拟环境并使用PowerShell兼容指令
"""

import os
import sys
import subprocess

def check_environment():
    """检查工作环境配置"""
    print("=== 环境检查 ===")
    
    # 1. 检查是否在虚拟环境中
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        print(f"✅ 虚拟环境已激活: {virtual_env}")
    else:
        print("❌ 虚拟环境未激活")
        print("请在PowerShell中运行: .\\dfp\\Scripts\\Activate.ps1")
        return False
    
    # 2. 检查操作系统
    if os.name == 'nt':  # Windows
        print("✅ Windows系统 - 使用PowerShell兼容指令")
    else:
        print("⚠️  非Windows系统")
    
    # 3. 检查关键模块
    try:
        import cv2
        import PIL
        import numpy as np
        print("✅ 关键模块已安装 (opencv, PIL, numpy)")
    except ImportError as e:
        print(f"❌ 缺少关键模块: {e}")
        return False
    
    return True

def powershell_compatible_command(command):
    """转换为PowerShell兼容的命令"""
    # 将Linux风格路径转换为Windows风格
    if isinstance(command, str):
        # 替换路径分隔符
        command = command.replace('/', '\\')
        # 替换常见的Linux命令
        command = command.replace('ls ', 'Get-ChildItem ')
        command = command.replace('cat ', 'Get-Content ')
        command = command.replace('grep ', 'Select-String ')
    
    return command

def run_with_environment_check(func):
    """装饰器：在运行前检查环境"""
    def wrapper(*args, **kwargs):
        if not check_environment():
            print("\n请先解决环境问题再运行脚本")
            return None
        return func(*args, **kwargs)
    return wrapper

# AI助手使用的标准命令模板
POWERSHELL_COMMANDS = {
    'list_files': 'Get-ChildItem',
    'check_file_exists': 'Test-Path "filename"',
    'get_content': 'Get-Content "filename"',
    'find_text': 'Select-String "pattern" "filename"',
    'activate_env': '.\\dfp\\Scripts\\Activate.ps1',
    'python_run': 'python script.py',
    'set_location': 'Set-Location "D:\\ws\\DeepFloorplan"'
}

if __name__ == "__main__":
    check_environment()
    
    print("\n=== PowerShell命令参考 ===")
    for name, cmd in POWERSHELL_COMMANDS.items():
        print(f"{name}: {cmd}")
