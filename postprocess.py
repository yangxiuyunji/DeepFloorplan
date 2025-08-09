import argparse
import os
import sys
import glob
import time
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import *

parser = argparse.ArgumentParser()

parser.add_argument('--result_dir', type=str, default='./out',
					help='The folder that save network predictions.')
parser.add_argument('--min_area', type=int, default=100,
					help='Minimum area threshold for room regions.')
parser.add_argument('--merge_boundary', action='store_true', default=True,
					help='Whether to merge boundary lines in the output.')
parser.add_argument('--verbose', action='store_true', default=False,
					help='Print detailed processing information.')

def imread(path, mode='RGB'):
	"""安全的图像读取函数"""
	try:
		return np.array(Image.open(path).convert(mode))
	except Exception as e:
		print(f"❌ 读取图像失败 {path}: {e}")
		return None

def imsave(path, array):
	"""安全的图像保存函数"""
	try:
		if array is None:
			print(f"❌ 无法保存空图像到 {path}")
			return False
		
		# 确保数组是正确的数据类型和范围
		if array.dtype == np.float64 or array.dtype == np.float32:
			# 如果是浮点数，确保在0-255范围内
			if array.max() <= 1.0:
				array = (array * 255).astype(np.uint8)
			else:
				array = np.clip(array, 0, 255).astype(np.uint8)
		elif array.dtype != np.uint8:
			array = np.clip(array, 0, 255).astype(np.uint8)
		
		# 处理维度问题
		if len(array.shape) == 3 and array.shape[0] == 1:
			array = array.squeeze(0)  # 移除第一个维度
		
		Image.fromarray(array).save(path)
		return True
	except Exception as e:
		print(f"❌ 保存图像失败 {path}: {e}")
		return False

def post_process(input_dir, save_dir, min_area=100, merge=True, verbose=False):
	"""
	后处理函数：精细化处理户型图分割结果
	
	Args:
		input_dir: 输入目录，包含网络预测结果
		save_dir: 输出目录
		min_area: 最小区域面积阈值
		merge: 是否合并边界线
		verbose: 是否打印详细信息
	"""
	# 创建输出目录
	Path(save_dir).mkdir(parents=True, exist_ok=True)
	
	# 获取所有PNG文件
	input_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
	
	if not input_paths:
		print(f"❌ 在目录 {input_dir} 中未找到PNG文件")
		return
	
	print(f"🔍 找到 {len(input_paths)} 个文件待处理")
	
	# 使用平台无关的路径处理
	names = [Path(p).name for p in input_paths]
	out_paths = [os.path.join(save_dir, name) for name in names]

	success_count = 0
	
	# 使用进度条
	for i, (input_path, output_path) in enumerate(tqdm(zip(input_paths, out_paths), 
													   desc="处理进度", 
													   total=len(input_paths))):
		try:
			# 读取图像
			im = imread(input_path, mode='RGB')
			if im is None:
				continue
				
			# 转换为索引图像
			im_ind = rgb2ind(im, color_map=floorplan_fuse_map)
			
			# 分离房间分割和边界分割
			rm_ind = im_ind.copy()
			rm_ind[im_ind==9] = 0   # 移除边界
			rm_ind[im_ind==10] = 0  # 移除边界

			bd_ind = np.zeros(im_ind.shape, dtype=np.uint8)
			bd_ind[im_ind==9] = 9   # 保留边界
			bd_ind[im_ind==10] = 10 # 保留边界

			hard_c = (bd_ind>0).astype(np.uint8)

			# 从房间预测创建区域掩码
			rm_mask = np.zeros(rm_ind.shape)
			rm_mask[rm_ind>0] = 1			
			
			# 从close_wall线创建区域		
			cw_mask = hard_c
			
			# 通过填补亮线之间的间隙来精细化close wall mask	
			cw_mask = fill_break_line(cw_mask)
				
			fuse_mask = cw_mask + rm_mask
			fuse_mask[fuse_mask>=1] = 255

			# 通过填补孔洞来精细化融合掩码
			fuse_mask = flood_fill(fuse_mask)
			fuse_mask = fuse_mask // 255	

			# 一个房间一个标签
			new_rm_ind = refine_room_region(cw_mask, rm_ind, min_area=min_area)

			# 忽略背景错误标记
			new_rm_ind = fuse_mask * new_rm_ind

			if verbose:
				print(f'📸 保存第{i+1}个精细化房间预测到 {output_path}')
			
			# 根据merge参数决定是否合并边界
			if merge:
				new_rm_ind[bd_ind==9] = 9
				new_rm_ind[bd_ind==10] = 10
				rgb = ind2rgb(new_rm_ind, color_map=floorplan_fuse_map)
			else:
				rgb = ind2rgb(new_rm_ind)
			
			# 保存结果
			if imsave(output_path, rgb):
				success_count += 1
				
		except Exception as e:
			print(f"❌ 处理文件 {input_path} 时发生错误: {e}")
			if verbose:
				import traceback
				traceback.print_exc()
			continue
	
	print(f"✅ 处理完成！成功处理 {success_count}/{len(input_paths)} 个文件")
	print(f"📁 结果保存在: {save_dir}")

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()

	input_dir = FLAGS.result_dir
	save_dir = os.path.join(input_dir, 'post')

	print("🚀 开始户型图后处理...")
	print(f"📂 输入目录: {input_dir}")
	print(f"📂 输出目录: {save_dir}")
	print(f"⚙️ 最小区域面积: {FLAGS.min_area}")
	print(f"⚙️ 合并边界: {'是' if FLAGS.merge_boundary else '否'}")
	
	try:
		post_process(input_dir, save_dir, 
					min_area=FLAGS.min_area,
					merge=FLAGS.merge_boundary,
					verbose=FLAGS.verbose)
	except Exception as e:
		print(f"❌ 后处理失败: {e}")
		if FLAGS.verbose:
			import traceback
			traceback.print_exc()
