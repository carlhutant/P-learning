import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
file_type = '.JPEG'
directory_path = "./data/AWA2/test_draw/"
# Generate tfrecord writer
result_tf_file = directory_path + '/test_draw_color_diff_121.tfrecords'
writer = tf.io.TFRecordWriter(result_tf_file)

for r, d, f in os.walk(directory_path):
    for dir in d: