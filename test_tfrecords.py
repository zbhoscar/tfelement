# -*- coding: UTF-8 -*- #
import os
import tensorflow as tf
import multiprocessing
import numpy as np
from PIL import Image



# def get_reshaped_gfile(pic_path, j, min_side=256):
#     pic_file = os.path.join(pic_path, j)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04\1.jpg
#     # resize images to min = 256
#     img = Image.open(pic_file)
#     new_size = [round(min_side * x / min(img.size)) for x in img.size]  # new_size for img_min_side=256
#     reshaped_img = img.resize(new_size)
#     reshaped_pic_file = os.path.join(pic_path, 'tmp_' + j)
#     reshaped_img.save(reshaped_pic_file, 'jpeg')
#     image_raw = tf.gfile.FastGFile(reshaped_pic_file, 'rb').read()
#     os.remove(reshaped_pic_file)
#     return image_raw
def get_reshaped_gfile(pic_path, j, min_side=256):
    pic_file = os.path.join(pic_path, j)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04\1.jpg
    # resize images to min = 256
    # img = Image.open(pic_file)
    image_raw = tf.gfile.FastGFile(pic_file, 'rb').read()
    image_orig = tf.image.decode_jpeg(image_raw)
    orig_size = image_orig.eval().shape[:2]
    new_size = [round(min_side * x / min(orig_size)) for x in orig_size]  # new_size for img_min_side=256#
    image = tf.image.convert_image_dtype(image_orig, dtype=tf.float32)
    image = tf.image.resize_images(image, new_size, method=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    encoded_image = tf.image.encode_jpeg(image, quality=50, optimize_size=True)
    output = encoded_image.eval()
    with tf.gfile.GFile(r'D:\Desktop\test.jpg', 'wb') as f:
        f.write(output)
    return encoded_image.eval()

def of_dict_reader(of_dict_file):
    f = open(of_dict_file, 'r')
    dict_str = f.read()
    of_dict = eval('{' + dict_str + '}')
    f.close()
    return of_dict





