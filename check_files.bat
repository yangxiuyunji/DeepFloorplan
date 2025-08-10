@echo off
chcp 65001 >nul
echo ===========================================
echo     DeepFloorplan 文件完整性检查
echo ===========================================
echo.

set MISSING_FILES=0

echo [检查核心程序文件]
if exist demo_fixed.py (echo ✅ demo_fixed.py) else (echo ❌ 缺少: demo_fixed.py & set /a MISSING_FILES+=1)
if exist demo.py (echo ✅ demo.py) else (echo ❌ 缺少: demo.py & set /a MISSING_FILES+=1)
if exist net_fixed.py (echo ✅ net_fixed.py) else (echo ❌ 缺少: net_fixed.py & set /a MISSING_FILES+=1)
if exist net.py (echo ✅ net.py) else (echo ❌ 缺少: net.py & set /a MISSING_FILES+=1)
if exist postprocess.py (echo ✅ postprocess.py) else (echo ❌ 缺少: postprocess.py & set /a MISSING_FILES+=1)
if exist preprocess.py (echo ✅ preprocess.py) else (echo ❌ 缺少: preprocess.py & set /a MISSING_FILES+=1)
if exist requirements.txt (echo ✅ requirements.txt) else (echo ❌ 缺少: requirements.txt & set /a MISSING_FILES+=1)
echo.

echo [检查预训练模型]
if exist pretrained (
    if exist pretrained\checkpoint (echo ✅ pretrained\checkpoint) else (echo ❌ 缺少: pretrained\checkpoint & set /a MISSING_FILES+=1)
    if exist pretrained\pretrained_r3d.data-00000-of-00001 (echo ✅ pretrained\pretrained_r3d.data-*) else (echo ❌ 缺少: pretrained\pretrained_r3d.data-* & set /a MISSING_FILES+=1)
    if exist pretrained\pretrained_r3d.index (echo ✅ pretrained\pretrained_r3d.index) else (echo ❌ 缺少: pretrained\pretrained_r3d.index & set /a MISSING_FILES+=1)
    if exist pretrained\pretrained_r3d.meta (echo ✅ pretrained\pretrained_r3d.meta) else (echo ❌ 缺少: pretrained\pretrained_r3d.meta & set /a MISSING_FILES+=1)
) else (
    echo ❌ 缺少: pretrained目录
    set /a MISSING_FILES+=4
)
echo.

echo [检查OCR模块]
if exist utils (
    if exist utils\ocr_enhanced.py (echo ✅ utils\ocr_enhanced.py) else (echo ❌ 缺少: utils\ocr_enhanced.py & set /a MISSING_FILES+=1)
) else (
    echo ❌ 缺少: utils目录
    set /a MISSING_FILES+=1
)
echo.

echo [检查测试数据]
if exist demo (
    if exist demo\demo.jpg (echo ✅ demo\demo.jpg) else (echo ❌ 缺少: demo\demo.jpg & set /a MISSING_FILES+=1)
    if exist demo\demo1.jpg (echo ✅ demo\demo1.jpg) else (echo ⚠️  建议添加: demo\demo1.jpg)
    if exist demo\demo2.jpg (echo ✅ demo\demo2.jpg) else (echo ⚠️  建议添加: demo\demo2.jpg)
    if exist demo\demo3.jpg (echo ✅ demo\demo3.jpg) else (echo ⚠️  建议添加: demo\demo3.jpg)
) else (
    echo ❌ 缺少: demo目录
    set /a MISSING_FILES+=1
)
echo.

echo [检查部署文件]
if exist setup.bat (echo ✅ setup.bat) else (echo ❌ 缺少: setup.bat & set /a MISSING_FILES+=1)
if exist 部署指南.md (echo ✅ 部署指南.md) else (echo ⚠️  建议添加: 部署指南.md)
if exist PaddleOCR配置优化指南.md (echo ✅ PaddleOCR配置优化指南.md) else (echo ⚠️  建议添加: PaddleOCR配置优化指南.md)
echo.

echo ===========================================
if %MISSING_FILES%==0 (
    echo           ✅ 检查通过！
    echo ===========================================
    echo.
    echo 🎉 所有必需文件都存在，可以开始部署！
    echo.
    echo 📋 下一步操作:
    echo   1. 运行 setup.bat 进行自动部署
    echo   2. 或者手动安装依赖包
    echo.
) else (
    echo           ❌ 检查失败！
    echo ===========================================
    echo.
    echo 😰 发现 %MISSING_FILES% 个缺失文件，请补充后再部署！
    echo.
    echo 📋 解决方案:
    echo   1. 重新下载完整的项目文件
    echo   2. 确保所有文件都已复制
    echo   3. 检查文件路径是否正确
    echo.
)

echo 📚 更多信息请查看: 部署指南.md
echo.
pause
