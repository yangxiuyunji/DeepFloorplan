@echo off
echo 正在激活dfp虚拟环境...
cd /d "C:\workspace\feng_shui\DeepFloorplan"
call dfp\Scripts\activate.bat
echo.
echo ===========================================
echo   DeepFloorplan 环境已激活！
echo   当前目录: %CD%
echo   Python版本: 
python --version
echo ===========================================
echo.
echo 可以运行以下命令测试项目：
echo   python demo.py
echo.
echo 按任意键继续...
pause > nul
cmd /k
