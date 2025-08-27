import os
import json
import time
import subprocess
from pathlib import Path

# 需要批量处理的文件名（位于 demo/ 目录下）
TARGET_IMAGES = [
    'demo1.jpg','demo2.jpg','demo3.jpg','demo4.jpg','demo5.jpg','demo6.jpg','demo7.jpg','demo8.jpg','demo9.jpg'
]

DEMO_DIR = Path('demo')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_SUMMARY = []

"""批量运行脚本：修复使用系统 python 导致依赖缺失的问题。
优先使用本项目虚拟环境 dfp/Scripts/python.exe。"""

venv_python = Path('dfp') / 'Scripts' / 'python.exe'
if venv_python.exists():
    python_exe = str(venv_python)
else:
    python_exe = 'python'  # 回退
    print('[警告] 未找到虚拟环境 python, 使用系统 python 可能出现依赖缺失 (如 numpy).')
main_script = 'demo_refactored_clean.py'

def run_one(image_name: str):
    img_path = DEMO_DIR / image_name
    if not img_path.exists():
        print(f'[跳过] {image_name} 不存在')
        return
    print(f'================ 处理 {image_name} ================')
    start = time.time()
    try:
        # 调用主脚本 (使用虚拟环境解释器)
        proc = subprocess.run([python_exe, main_script, str(img_path)], capture_output=True, text=True, timeout=900)
        duration = time.time() - start
        # 保存单独日志
        log_file = OUTPUT_DIR / f'{img_path.stem}_run.log'
        log_file.write_text(proc.stdout + '\n---- STDERR ----\n' + proc.stderr, encoding='utf-8', errors='ignore')
        if proc.returncode != 0:
            # 提取首行错误指示
            first_err_line = next((l for l in proc.stderr.splitlines() if 'Error' in l or 'ModuleNotFoundError' in l), '').strip()
            if first_err_line:
                print(f'[错误] {image_name}: {first_err_line}')
        print(f'[完成] {image_name} 用时 {duration:.1f}s, 日志 -> {log_file.name}')
        # 尝试读取结构化结果JSON
        result_json = OUTPUT_DIR / f'{img_path.stem}_result.json'
        summary = {
            'image': image_name,
            'time_sec': round(duration,2),
            'status': 'ok' if proc.returncode==0 else f'err({proc.returncode})',
            'interpreter': python_exe
        }
        if result_json.exists():
            try:
                data = json.loads(result_json.read_text(encoding='utf-8', errors='ignore'))
                # 简要提取统计
                summary.update({
                    'rooms_detected': len(data.get('rooms', [])),
                    'has_kitchen': any(r.get('type')=='厨房' for r in data.get('rooms', [])),
                    'bathrooms': sum(1 for r in data.get('rooms', []) if '卫' in r.get('name','') or r.get('type')=='卫生间'),
                })
            except Exception as e:
                summary['json_error'] = str(e)
        else:
            summary['json_missing'] = True
        RESULTS_SUMMARY.append(summary)
    except subprocess.TimeoutExpired:
        print(f'[超时] {image_name} 超过时间限制，跳过')
        RESULTS_SUMMARY.append({'image': image_name,'status':'timeout'})


def main():
    print('批量户型识别开始...')
    for img in TARGET_IMAGES:
        run_one(img)
    # 汇总输出
    summary_path = OUTPUT_DIR / 'batch_summary.json'
    summary_path.write_text(json.dumps(RESULTS_SUMMARY, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n批量完成，汇总:')
    for item in RESULTS_SUMMARY:
        print(item)
    print(f'汇总文件: {summary_path}')

if __name__ == '__main__':
    main()
