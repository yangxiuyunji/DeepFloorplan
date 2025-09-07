# AI助手代码生成规范
# AI Assistant Code Generation Standards

## 🛡️ 环境检查规范
每次运行Python命令前必须确保：
1. ✅ 虚拟环境已激活: `$env:VIRTUAL_ENV`
2. ✅ 使用PowerShell兼容语法

## 📁 文件组织规范

### 临时文件命名和存放
所有AI生成的临时文件必须放在 `debug/` 文件夹下：

```
debug/
├── debug_*.py          # 调试脚本
├── test_*.py           # 测试脚本  
├── verify_*.py         # 验证脚本
├── analyze_*.py        # 分析脚本
├── temp_*.py           # 临时脚本
└── prototype_*.py      # 原型脚本
```

### 命名规范
- **debug_xxx.py**: 用于调试特定功能的脚本
- **test_xxx.py**: 单元测试或功能测试脚本
- **verify_xxx.py**: 验证计算结果或配置的脚本
- **analyze_xxx.py**: 数据分析或统计脚本
- **temp_xxx.py**: 临时性质的一次性脚本
- **prototype_xxx.py**: 功能原型验证脚本

### 永久文件标准
只有确认稳定且有长期价值的代码才可以放在根目录：
- 主要功能模块
- 核心工具脚本
- 配置文件
- 文档文件

## 🔧 PowerShell命令标准

### 文件操作
```powershell
# 正确 ✅
Test-Path "debug\test_file.py"
New-Item -ItemType Directory -Path "debug" -Force
Move-Item "temp_*.py" -Destination "debug\" -Force
Get-ChildItem "debug\" -Filter "*.py"

# 错误 ❌
[ -f "debug/test_file.py" ]
mkdir -p debug
mv temp_*.py debug/
ls debug/*.py
```

### 路径分隔符
```powershell
# 正确 ✅ (Windows PowerShell支持两种)
.\debug\test_file.py
./debug/test_file.py    # 也支持

# 建议使用 Windows 风格 (更明确)
.\debug\test_file.py
```

### 虚拟环境检查
```powershell
# 标准检查模式
if (-not $env:VIRTUAL_ENV) { 
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\dfp\Scripts\Activate.ps1 
}
```

## 🐍 Python脚本模板

### 临时调试脚本模板
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时调试脚本: [功能描述]
生成时间: [日期]
目的: [具体调试目标]
"""

import os
import sys

# 环境检查
if not os.environ.get('VIRTUAL_ENV'):
    print("警告: 虚拟环境未激活")
    print("请在PowerShell中运行: .\\dfp\\Scripts\\Activate.ps1")
    sys.exit(1)

def main():
    """主函数"""
    print(f"=== {__file__} 调试开始 ===")
    
    # 调试代码在这里
    
    print("=== 调试完成 ===")

if __name__ == "__main__":
    main()
```

### 验证脚本模板
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证脚本: [验证内容]
"""

def verify_xxx():
    """验证特定功能"""
    try:
        # 验证逻辑
        print("✅ 验证通过")
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_xxx()
    sys.exit(0 if success else 1)
```

## 📝 AI生成代码检查清单

在生成任何代码前，AI助手必须确认：

- [ ] 虚拟环境已激活 (`$env:VIRTUAL_ENV`)
- [ ] 使用PowerShell兼容语法
- [ ] 临时文件放在 `debug/` 文件夹
- [ ] 文件命名符合规范 (`debug_*`, `test_*`, `verify_*` 等)
- [ ] 路径使用 Windows 反斜杠风格 `.\debug\file.py`
- [ ] 包含适当的错误检查和用户提示

## 🔄 清理命令

定期清理临时文件：
```powershell
# 清理所有调试文件 (慎用)
Remove-Item "debug\debug_*.py" -Force
Remove-Item "debug\temp_*.py" -Force

# 查看debug文件夹内容
Get-ChildItem "debug\" | Sort-Object Name
```

这样可以保持项目目录整洁，同时方便管理临时文件。
