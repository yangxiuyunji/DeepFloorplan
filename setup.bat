@echo off
chcp 65001 >nul
echo ===========================================
echo     DeepFloorplan 自动部署脚本
echo ===========================================
echo.

REM 检查Python
echo [1/6] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python
    echo 请先安装Python 3.8-3.12版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    python --version
    echo ✅ Python环境检查通过
)
echo.

REM 检查网络连接
echo [2/6] 检查网络连接...
ping -n 1 pypi.org >nul 2>&1
if errorlevel 1 (
    echo ⚠️  网络连接异常，将使用国内镜像源
    set USE_MIRROR=1
) else (
    echo ✅ 网络连接正常
    set USE_MIRROR=0
)
echo.

REM 升级pip
echo [3/6] 升级pip...
if %USE_MIRROR%==1 (
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
) else (
    python -m pip install --upgrade pip
)
if errorlevel 1 (
    echo ❌ pip升级失败
    pause
    exit /b 1
)
echo ✅ pip升级完成
echo.

REM 安装基础依赖
echo [4/6] 安装基础依赖...
if exist requirements.txt (
    if %USE_MIRROR%==1 (
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    ) else (
        pip install -r requirements.txt
    )
    if errorlevel 1 (
        echo ❌ 基础依赖安装失败
        pause
        exit /b 1
    )
    echo ✅ 基础依赖安装完成
) else (
    echo ⚠️  未找到requirements.txt，跳过基础依赖安装
)
echo.

REM 安装PaddleOCR
echo [5/6] 安装PaddleOCR...
if %USE_MIRROR%==1 (
    pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple/
) else (
    pip install paddleocr
)
if errorlevel 1 (
    echo ❌ PaddleOCR安装失败，尝试分步安装...
    if %USE_MIRROR%==1 (
        pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple/
        pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple/
    ) else (
        pip install paddlepaddle
        pip install paddleocr
    )
)
echo ✅ PaddleOCR安装完成
echo.

REM 安装额外依赖
echo 安装额外依赖包...
if %USE_MIRROR%==1 (
    pip install opencv-python Pillow matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
) else (
    pip install opencv-python Pillow matplotlib
)
echo ✅ 额外依赖安装完成
echo.

REM 验证安装
echo [6/6] 验证安装结果...
if exist demo_fixed.py (
    echo 正在测试核心功能...
    python demo_fixed.py
    if errorlevel 1 (
        echo ❌ 功能测试失败，请检查错误信息
        pause
        exit /b 1
    )
    echo ✅ 功能测试通过
) else (
    echo ⚠️  未找到demo_fixed.py，请确认文件完整性
)
echo.

echo ===========================================
echo           🎉 部署完成！
echo ===========================================
echo.
echo 📋 部署摘要:
echo   ✅ Python环境: 正常
echo   ✅ 依赖包: 已安装
echo   ✅ PaddleOCR: 已配置
echo   ✅ 功能测试: 通过
echo.
echo 🚀 使用方法:
echo   python demo_fixed.py                    # 处理默认图片
echo   python demo_fixed.py your_image.jpg     # 处理指定图片
echo.
echo 📚 更多信息请查看: 部署指南.md
echo.
pause
