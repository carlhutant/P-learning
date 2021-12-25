import configure
import os
import cv2
import numpy as np
from pathlib import Path


# Dataset config #
dataset = 'AWA2'  # AWA2, imagenet
datatype = 'img'  # img, tfrecord, npy
data_advance = 'color_diff_121_abs_3ch'   # color_diff_121, none, color_diff_121_abs
data_usage = 'train'

# Directory set
dataset_dir = configure.dataset_dir
target_directory = '{}/{}/{}/{}/{}/'.format(dataset_dir, dataset, datatype, data_advance, data_usage)

walk_generator = os.walk(target_directory)
root, directory, _ = next(walk_generator)
total_sum = np.zeros(configure.channel, dtype=np.int64)
pixel_count = 0
for d in directory:
    print(d)
    walk_generator2 = os.walk(root + d)
    flies_root, _, files = next(walk_generator2)
    for file in files:
        array = cv2.imread(str(Path(flies_root).joinpath(file)))
        height, width, _ = array.shape
        pixel_count = pixel_count + height * width
        array = array.sum(0, dtype=np.int64)
        array = array.sum(0, dtype=np.int64)
        total_sum = total_sum + array
total_sum = total_sum/pixel_count
a = 0
