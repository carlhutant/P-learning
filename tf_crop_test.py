import numpy as np
import tensorflow as tf
import cv2
BATCH_SIZE = 1
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
CROP_SIZE = (24, 24)

img = cv2.imread('E:/Dataset/AWA2/img/none/test/chimpanzee/chimpanzee_10001.jpg')
# img2 = cv2.imread('E:/Dataset/AWA2/img/none/test/chimpanzee/chimpanzee_10002.jpg')
# img = img1[np.newaxis, ...]
# img2 = img2[np.newaxis, ...]
# img = np.concatenate((img1, img2), 0)
tf_img = tf.convert_to_tensor(img, dtype=tf.float32)

output = tf.image.random_crop(tf_img, (224, 224, 3), 486)

img2 = output.numpy()
img2 = np.array(img2, dtype=np.uint8)
cv2.imshow('123', img2)
cv2.waitKey()
