@echo off
echo =====================================
echo   DeepFloorplan 厨房识别系统
echo =====================================
echo.
cd /d "C:\workspace\feng_shui\DeepFloorplan"
call dfp\Scripts\activate.bat
echo 环境已激活！
echo.
echo 🎯 可用命令:
echo   1. 基本识别: python demo.py --im_path=./demo/45765448.jpg
echo   2. 测试脚本: python test_kitchen.py  
echo   3. 查看颜色说明: python color_guide.py
echo.
echo 🍳 厨房识别功能特点:
echo   • 独立识别厨房区域 (橙黄色显示)
echo   • 支持中英文房间标注识别
echo   • 基于AI分割 + OCR + 空间分析
echo.
echo =====================================
cmd /k
