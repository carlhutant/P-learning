import os
import numpy as np
import cv2

# file_type = '.jpg'
directory_path = 'E:/Dataset/imagenet/tfrecord/color_diff_121/'
for r, d, f in os.walk(directory_path):
    for file in f:
        # if file.endswith(file_type):
        if os.path.splitext(file)[0].endswith("_train"):
            origin_file_name = os.path.splitext(os.path.join(r, file))
            new_file_name = r + 'train' + origin_file_name[1]
            os.rename(os.path.join(r, file), new_file_name)
