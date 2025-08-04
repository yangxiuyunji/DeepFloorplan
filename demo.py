import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf

# Disable TF 2.x behavior for compatibility
tf.disable_v2_behavior()

from PIL import Image
import numpy as np

# OCR utilities
from utils.ocr import extract_room_text, fuse_ocr_and_segmentation

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
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
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
from matplotlib import pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Force CPU usage - disable GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def main(args):
        # load input
        im = imread(args.im_path, mode='RGB')
        # Resize image for network and OCR. Keep uint8 copy for OCR
        im = imresize(im, (512, 512))
        ocr_im = im.copy()
        # Extract textual room labels using OCR
        ocr_results = extract_room_text(ocr_im)
        # Convert to float and normalize for network inference
        im = im.astype(np.float32) / 255.

        # create tensorflow session with CPU configuration
        config = tf.ConfigProto(
                device_count={'GPU': 0},  # Disable GPU
                allow_soft_placement=True,
                log_device_placement=False
        )
        with tf.Session(config=config) as sess:
                
                # initialize
                sess.run(tf.group(tf.global_variables_initializer(),
                                        tf.local_variables_initializer()))

                # restore pretrained model
                saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
                saver.restore(sess, './pretrained/pretrained_r3d')

                # get default graph
                graph = tf.get_default_graph()

                # restore inputs & outpus tensor
                x = graph.get_tensor_by_name('inputs:0')
                room_type_logit = graph.get_tensor_by_name('Cast:0')
                room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

                # infer results
                [room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
                                                                feed_dict={x:im.reshape(1,512,512,3)})
                room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

                # merge results
                floorplan = room_type.copy()
                floorplan[room_boundary==1] = 9
                floorplan[room_boundary==2] = 10
                # Use OCR labels to refine room categories
                floorplan = fuse_ocr_and_segmentation(floorplan, ocr_results)
                floorplan_rgb = ind2rgb(floorplan)

                # plot results
                plt.subplot(121)
                plt.imshow(im)
                plt.subplot(122)
                plt.imshow(floorplan_rgb/255.)
                plt.show()

if __name__ == '__main__':
        FLAGS, unparsed = parser.parse_known_args()
        main(FLAGS)
