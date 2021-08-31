import os
import numpy as np
import cv2

file_type = '.jpg'
directory_path = "D:/Program/Python/p-learning/P_learning/data/AWA2/test_rename"
for r, d, f in os.walk(directory_path):
    for file in f:
        if file.endswith(file_type):
            if os.path.splitext(file)[0].endswith("_edge"):
                origin_file_name = os.path.splitext(os.path.join(r, file))
                new_file_name = origin_file_name[0] + '_extend' + origin_file_name[1]
                os.rename(os.path.join(r, file), new_file_name)
