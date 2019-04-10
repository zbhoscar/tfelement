import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))
img_raw = cv2.imread("1.jpg")
img_flt = img_raw.astype(np.float32) / 255
image  = img_flt

# cv2.startWindowThread()
cv2.namedWindow('image')
cv2.imshow('image', image)
cv2.waitKey(3000)
cv2.destroyWindow("image")
# cv2.destroyAllWindows()

# image = tf.image.random_saturation(image, 0.5, 1.5)
# image = tf.image.random_brightness(image, 32. / 255.)
# image = tf.image.random_hue(image, 0.2)
# image = tf.image.random_contrast(image, 0.5, 1.5)
# image = tf.clip_by_value(image, 0.0, 1.0)

image = tf.random_crop(image, [224,224,3])
sess=tf.Session()
t = sess.run(image)
plt.imshow(t[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
sess.close()


plt.figure("Image")  # 图像窗口名称
plt.imshow(img)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('image')  # 图像题目
plt.show()



img = Image.open('1.jpg')
gray = img.convert('L')
r, g, b = img.split()
img_merged = Image.merge('RGB', (r, g, b))

plt.figure('test')  # 设置窗口大小
plt.suptitle('Multi_Image')  # 图片名称
plt.subplot(2, 3, 1), plt.title('image')
plt.imshow(img), plt.axis('off')
plt.subplot(2, 3, 2), plt.title('gray')
plt.imshow(gray, cmap='gray'), plt.axis('off')  # 这里显示灰度图要加cmap
plt.subplot(2, 3, 3), plt.title('img_merged')
plt.imshow(img_merged), plt.axis('off')
plt.subplot(2, 3, 4), plt.title('r')
plt.imshow(r, cmap='gray'), plt.axis('off')
plt.subplot(2, 3, 5), plt.title('g')
plt.imshow(g, cmap='gray'), plt.axis('off')
plt.subplot(2, 3, 6), plt.title('b')
plt.imshow(b, cmap='gray'), plt.axis('off')
plt.show()
plt.savefig('test' + '.png')


import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('1.jpg','rb').read()

with tf.Session() as sess:
     img_data = tf.image.decode_jpeg(image_raw_data)
     plt.imshow(img_data.eval())
     plt.show()

     image = tf.random_crop(img_data, [224, 224, 3])
     image = tf.image.random_flip_left_right(image)
     image = tf.image.random_brightness(image, max_delta=63. / 255.)
     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
     image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
     image = tf.image.per_image_standardization(image)
     plt.imshow(image.eval())
     plt.show()


     adjusted = tf.image.adjust_brightness(img_data, -0.5)
     plt.imshow(adjusted.eval())
     plt.show()
     adjusted = tf.image.adjust_brightness(img_data, 0.5)
     plt.imshow(adjusted.eval())
     plt.show()

     adjusted = tf.image.adjust_contrast(img_data, -5)
     plt.imshow(adjusted.eval())
     plt.show()
     adjusted = tf.image.adjust_contrast(img_data, 5)
     plt.imshow(adjusted.eval())
     plt.show()

     adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
     plt.imshow(adjusted.eval())
     plt.show()

     adjusted = tf.image.random_contrast(img_data, 0.1, 0.6)
     plt.imshow(adjusted.eval())

     plt.show()
     adjusted = tf.image.adjust_hue(img_data, 0.1)
     plt.imshow(adjusted.eval())
     plt.show()

     adjusted = tf.image.random_hue(img_data, 0.5)
     plt.imshow(adjusted.eval())
     plt.show()  


     adjusted = tf.image.adjust_saturation(img_data, -5)
     plt.imshow(adjusted.eval())
     plt.show()  


     adjusted = tf.image.random_saturation(img_data, 0, 5)
     plt.imshow(adjusted.eval())
     plt.show()  

     adjusted = tf.image.per_image_standardization(img_data)
     plt.imshow(adjusted.eval())
     plt.show()  
