@echo off
echo 正在激活DeepFloorplan虚拟环境（dfp）...
cd /d d:\ws\DeepFloorplan
call dfp\Scripts\activate.bat
echo.
echo 环境已激活！当前工作目录: %CD%
echo 使用以下命令运行项目:
echo python demo.py --im_path=./demo/45719584.jpg
echo.
cmd /k
