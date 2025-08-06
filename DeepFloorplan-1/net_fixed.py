import numpy as np
import tensorflow.compat.v1 as tf # using tf 2.x with v1 compatibility

# Disable TF 2.x behavior for compatibility
tf.disable_v2_behavior()

import os
import sys
import glob
import time
import random

from scipy import ndimage
from PIL import Image

def imread(path, mode='RGB'):
    """Read image using PIL"""
    img = Image.open(path)
    if mode == 'RGB':
        img = img.convert('RGB')
    elif mode == 'L':
        img = img.convert('L')
    return np.array(img)

def imsave(path, img):
    """Save image using PIL"""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def imresize(img, size):
    """Resize image using PIL"""
    if len(img.shape) == 3:
        h, w, c = size if len(size) == 3 else (*size, img.shape[2])
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((w, h))
        return np.array(img_resized)
    else:
        h, w = size
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((w, h))
        return np.array(img_resized)

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import fast_hist
from tf_record import read_record, read_bd_rm_record

GPU_ID = '0'

def data_loader_bd_rm_from_tfrecord(batch_size=1):
	paths = open('./dataset/r3d_train.txt', 'r').read().splitlines()

	loader_dict = read_bd_rm_record('./dataset/r3d.tfrecords', batch_size=batch_size, size=512)

	num_batch = len(paths) // batch_size

	return loader_dict, num_batch

class Network(object):
	"""docstring for Network"""
	def __init__(self, dtype=tf.float32):
		print('Initial nn network object...')
		self.dtype = dtype
		self.pre_train_restore_map = {'vgg_16/conv1/conv1_1/weights':'FNet/conv1_1/W', # {'checkpoint_scope_var_name':'current_scope_var_name'} shape must be the same
									'vgg_16/conv1/conv1_1/biases':'FNet/conv1_1/b',	
									'vgg_16/conv1/conv1_2/weights':'FNet/conv1_2/W',
									'vgg_16/conv1/conv1_2/biases':'FNet/conv1_2/b',	
									'vgg_16/conv2/conv2_1/weights':'FNet/conv2_1/W',
									'vgg_16/conv2/conv2_1/biases':'FNet/conv2_1/b',	
									'vgg_16/conv2/conv2_2/weights':'FNet/conv2_2/W',
									'vgg_16/conv2/conv2_2/biases':'FNet/conv2_2/b',	
									'vgg_16/conv3/conv3_1/weights':'FNet/conv3_1/W',
									'vgg_16/conv3/conv3_1/biases':'FNet/conv3_1/b',	
									'vgg_16/conv3/conv3_2/weights':'FNet/conv3_2/W',
									'vgg_16/conv3/conv3_2/biases':'FNet/conv3_2/b',	
									'vgg_16/conv3/conv3_3/weights':'FNet/conv3_3/W',
									'vgg_16/conv3/conv3_3/biases':'FNet/conv3_3/b',	
									'vgg_16/conv4/conv4_1/weights':'FNet/conv4_1/W',
									'vgg_16/conv4/conv4_1/biases':'FNet/conv4_1/b',	
									'vgg_16/conv4/conv4_2/weights':'FNet/conv4_2/W',
									'vgg_16/conv4/conv4_2/biases':'FNet/conv4_2/b',	
									'vgg_16/conv4/conv4_3/weights':'FNet/conv4_3/W',
									'vgg_16/conv4/conv4_3/biases':'FNet/conv4_3/b',	
									'vgg_16/conv5/conv5_1/weights':'FNet/conv5_1/W',
									'vgg_16/conv5/conv5_1/biases':'FNet/conv5_1/b',	
									'vgg_16/conv5/conv5_2/weights':'FNet/conv5_2/W',
									'vgg_16/conv5/conv5_2/biases':'FNet/conv5_2/b',	
									'vgg_16/conv5/conv5_3/weights':'FNet/conv5_3/W',
									'vgg_16/conv5/conv5_3/biases':'FNet/conv5_3/b'} 

	# basic layer 
	def _he_uniform(self, shape, regularizer=None, trainable=None, name=None):
		name = 'W' if name is None else name+'/W'

		# size = (k_h, k_w, in_dim, out_dim)
		kernel_size = np.prod(shape[:2]) # k_h*k_w
		fan_in = shape[-2]*kernel_size  # fan_out = shape[-1]*kernel_size

		# compute the scale value
		s = np.sqrt(1. /fan_in)

		# create variable and specific GPU device
		with tf.device('/device:GPU:'+GPU_ID):
			w = tf.get_variable(name, shape, dtype=self.dtype,
							initializer=tf.random_uniform_initializer(minval=-s, maxval=s),
							regularizer=regularizer, trainable=trainable)

		return w

	# ... (rest of the Network class methods would need to be copied and fixed)
	
	def forward(self, x, init_with_pretrain_vgg=False):
		"""Forward pass - simplified version for demo"""
		# This is a placeholder implementation
		# The full implementation would need the complete network architecture
		pass
