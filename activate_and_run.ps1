# DeepFloorplan PowerShell 自动化脚本
# 激活虚拟环境并运行Python脚本

param(
    [string]$ScriptName = "luoshu_visualizer.py",
    [string]$JsonFile = ".\output\demo1_result_edited.json",
    [switch]$SkipEnvironmentCheck
)

Write-Host "=== DeepFloorplan 执行环境检查 ===" -ForegroundColor Green

# 1. 检查当前目录
$CurrentPath = Get-Location
Write-Host "当前目录: $CurrentPath"

if (-not (Test-Path "luoshu_visualizer.py")) {
    Write-Error "错误: 未在DeepFloorplan项目目录中"
    Write-Host "请先切换到项目目录: Set-Location 'D:\ws\DeepFloorplan'"
    exit 1
}

# 2. 激活虚拟环境
Write-Host "激活虚拟环境..." -ForegroundColor Yellow

if (Test-Path ".\dfp\Scripts\Activate.ps1") {
    # 激活虚拟环境
    & ".\dfp\Scripts\Activate.ps1"
    
    # 验证激活状态
    if (-not $env:VIRTUAL_ENV) {
        Write-Error "虚拟环境激活失败"
        exit 1
    }
    
    Write-Host "✅ 虚拟环境已激活: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Error "未找到虚拟环境: .\dfp\Scripts\Activate.ps1"
    Write-Host "请先创建虚拟环境或检查路径"
    exit 1
}

# 3. 验证Python环境
if (-not $SkipEnvironmentCheck) {
    Write-Host "验证Python环境..." -ForegroundColor Yellow
    
    $PythonVersion = python --version 2>&1
    Write-Host "Python版本: $PythonVersion"
    
    # 检查关键包
    $RequiredPackages = @("opencv-python", "Pillow", "numpy")
    foreach ($Package in $RequiredPackages) {
        $PackageInfo = pip list | Select-String $Package
        if ($PackageInfo) {
            Write-Host "✅ $Package 已安装" -ForegroundColor Green
        } else {
            Write-Warning "⚠️  $Package 未安装"
        }
    }
}

# 4. 运行Python脚本
Write-Host "执行脚本: python $ScriptName $JsonFile" -ForegroundColor Cyan

if (Test-Path $ScriptName) {
    if ($JsonFile -and (Test-Path $JsonFile)) {
        python $ScriptName $JsonFile
    } elseif ($JsonFile) {
        Write-Error "JSON文件不存在: $JsonFile"
        exit 1
    } else {
        python $ScriptName
    }
} else {
    Write-Error "Python脚本不存在: $ScriptName"
    exit 1
}

Write-Host "=== 执行完成 ===" -ForegroundColor Green
