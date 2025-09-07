# DeepFloorplan 工作环境配置指南

## 重要提醒事项

### 1. 虚拟环境激活 (必须)
```powershell
# 激活虚拟环境 (PowerShell)
.\dfp\Scripts\Activate.ps1

# 验证激活状态
$env:VIRTUAL_ENV
python --version
```

### 2. PowerShell 兼容指令

#### 文件路径操作
```powershell
# 正确 ✅
Get-ChildItem .\output\
Test-Path ".\demo\demo1.jpg"

# 错误 ❌ (Linux风格)
ls ./output/
[ -f "./demo/demo1.jpg" ]
```

#### 条件判断
```powershell
# 正确 ✅
if (Test-Path "file.json") { Write-Host "文件存在" }

# 错误 ❌ (Bash风格)
if [ -f "file.json" ]; then echo "文件存在"; fi
```

#### 变量操作
```powershell
# 正确 ✅
$env:PYTHONPATH = "D:\ws\DeepFloorplan"
Write-Host "Path: $env:PYTHONPATH"

# 错误 ❌ (Bash风格)
export PYTHONPATH="/d/ws/DeepFloorplan"
echo "Path: $PYTHONPATH"
```

#### 多命令执行
```powershell
# 正确 ✅
.\dfp\Scripts\Activate.ps1; python luoshu_visualizer.py .\output\demo1_result_edited.json

# 错误 ❌ (Bash风格)
source dfp/Scripts/activate && python luoshu_visualizer.py ./output/demo1_result_edited.json
```

## 标准工作流程

### 启动环境
```powershell
# 1. 进入项目目录
Set-Location "D:\ws\DeepFloorplan"

# 2. 激活虚拟环境
.\dfp\Scripts\Activate.ps1

# 3. 验证环境
python --version
pip list | Select-String "opencv\|pillow\|numpy"
```

### 主要功能脚本
```powershell
# 房间检测 (重构版 - 推荐)
python demo_refactored_clean.py demo\demo1.jpg

# 房间检测 (原版)
python demo.py demo\demo1.jpg

# 生成风水分析图
python luoshu_visualizer.py .\output\demo1_result_edited.json

# 打开房间编辑器
python editor\main.py

# 批量处理所有demo
python batch_run_demos.py
```

### 文件组织规范
```
DeepFloorplan/
├── 主要脚本/
│   ├── demo_refactored_clean.py  # 房间检测(重构版,推荐)
│   ├── demo.py                   # 房间检测(原版)
│   ├── luoshu_visualizer.py      # 风水分析可视化
│   └── batch_run_demos.py        # 批量处理
├── 工具脚本/
│   ├── quick_start.ps1           # 快速环境检查
│   ├── activate_and_run.ps1      # 自动化运行
│   └── environment_checker.py    # Python环境检查
└── debug/                        # 临时和调试文件
    ├── debug_*.py               # 调试脚本
    ├── test_*.py                # 测试脚本
    └── verify_*.py              # 验证脚本
```

### 常用PowerShell命令对照表

| 功能 | PowerShell | Linux/Bash |
|------|-----------|-------------|
| 列出文件 | `Get-ChildItem` 或 `ls` | `ls` |
| 检查文件存在 | `Test-Path "file"` | `[ -f "file" ]` |
| 创建目录 | `New-Item -ItemType Directory -Path "dir"` | `mkdir dir` |
| 复制文件 | `Copy-Item "src" "dest"` | `cp src dest` |
| 移动文件 | `Move-Item "src" "dest"` | `mv src dest` |
| 查看内容 | `Get-Content "file"` | `cat file` |
| 查找文本 | `Select-String "pattern" "file"` | `grep pattern file` |
| 环境变量 | `$env:VAR = "value"` | `export VAR="value"` |
| 路径分隔符 | `\` 或 `/` (都支持) | `/` |

## 自动化脚本模板

### 激活环境并运行 (PowerShell)
```powershell
# activate_and_run.ps1
param(
    [string]$ScriptName = "luoshu_visualizer.py",
    [string]$JsonFile = ".\output\demo1_result_edited.json"
)

# 激活虚拟环境
& ".\dfp\Scripts\Activate.ps1"

# 验证环境
if (-not $env:VIRTUAL_ENV) {
    Write-Error "虚拟环境激活失败"
    exit 1
}

# 运行Python脚本
python $ScriptName $JsonFile
```

### 使用方法
```powershell
# 运行特定文件
.\activate_and_run.ps1 -ScriptName "luoshu_visualizer.py" -JsonFile ".\output\demo2_result_edited.json"

# 使用默认参数
.\activate_and_run.ps1
```
