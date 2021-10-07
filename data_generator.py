import os
import cv2
import math
import random
import numpy as np
from configure import *


# 考慮 shuffle every epoch
def load_generator(target_directory, shuffle, shuffle_every_epoch):
    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    instance_list = []
    class_count = 0
    for d in directory:
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        for file in files:
            instance_list.append({'path': os.path.join(flies_root, file), 'label': class_count})
        class_count = class_count + 1
    file_num = len(instance_list)
    if shuffle or shuffle_every_epoch:
        random.shuffle(instance_list)
    yield file_num, class_count
    while True:
        for i in range(file_num):
            instance = instance_list[i]
            if datatype == 'img':
                img = cv2.imread(instance['path'])
                if color_mode == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif datatype == 'npy':
                img = np.load(instance['path'])
                if color_mode == "BGR":
                    img = img[..., [4, 5, 2, 3, 0, 1]]
            else:
                raise RuntimeError
            img = np.array(img, dtype=np.float32)
            label = np.zeros(class_num, dtype=np.float32)
            label[instance['label']] = 1
            yield img, label
        if shuffle_every_epoch:
            random.shuffle(instance_list)


# 考慮加入 last batch 捨去功能, rotate
def crop_generator(target_directory, batch_size, final_batch_opt, crop_type, crop_h, crop_w, resize_short_edge_max,
                   resize_short_edge_min, horizontal_flip, shuffle, shuffle_every_epoch):
    img_gen = load_generator(target_directory, shuffle, shuffle_every_epoch)
    file_num, dir_num = next(img_gen)
    print("Found {} images belonging to {} classes.".format(file_num, dir_num))
    if file_num != train_cardinality and file_num != val_cardinality:
        raise RuntimeError
    if dir_num != class_num:
        raise RuntimeError
    # if crop_type == 'ten_crop' and batch_size != 1:
    #     raise RuntimeError
    yield
    random.seed(random_seed)
    file_remain_num = file_num
    while True:
        batch_feature = np.empty([0, crop_h, crop_w, channel], dtype=np.float32)
        batch_label = np.empty([0, class_num], dtype=np.float32)
        if final_batch_opt == 'complete':
            batch_data_num = min(batch_size, file_remain_num)
        elif final_batch_opt == 'full':
            batch_data_num = batch_size
        else:
            print('final_batch_opt error.')
            raise RuntimeError
        for i in range(batch_data_num):
            img, label = next(img_gen)
            label = label[np.newaxis, ...]
            if horizontal_flip:
                if random.randint(0, 1):
                    img = img[:, ::-1, :]
            height, width, _ = img.shape
            if crop_type == 'none':
                new_height = crop_h
                new_width = crop_w
            elif height < width:
                new_height = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_width = round(width * new_height / height)
            else:
                new_width = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_height = round(height * new_width / width)
            img = cv2.resize(img, (new_width, new_height))

            y0_list = []
            x0_list = []
            if crop_type == 'none':
                y0_list.append(0)
                x0_list.append(0)
            elif crop_type == 'center':
                center_y = math.ceil(new_height / 2)
                center_x = math.ceil(new_width / 2)
                y0_list.append(center_y - math.ceil(crop_h / 2) + 1)
                x0_list.append(center_x - math.ceil(crop_w / 2) + 1)
            elif crop_type == 'random':  # 先不處理 crop 大於原圖的情況
                y0_list.append(random.randint(0, new_height - crop_h))
                x0_list.append(random.randint(0, new_width - crop_w))
            elif crop_type == 'ten_crop':
                center_y = math.ceil(new_height / 2)
                center_x = math.ceil(new_width / 2)
                y0_list.append(center_y - math.ceil(crop_h / 2) + 1)
                x0_list.append(center_x - math.ceil(crop_w / 2) + 1)
                y0_list.extend([0, 0, new_height - crop_h, new_height - crop_h])
                x0_list.extend([0, new_width - crop_h, 0, new_width - crop_h])
            else:
                raise RuntimeError
            crop_list = []
            for y0, x0 in zip(y0_list, x0_list):
                crop = img[y0:y0 + crop_h, x0:x0 + crop_w, :]
                crop_list.append(crop)
                if crop_type == 'ten_crop':
                    crop_list.append(crop[:, ::-1, :])
            for crop in crop_list:
                if preprocess == 'caffe':
                    if color_mode == 'BGR':
                        crop = crop - np.array([[[103.939, 116.779, 123.68]]], dtype=np.float32)
                    elif color_mode == 'RGB':
                        crop = crop - np.array([123.68, 116.779, 103.939], dtype=np.float32)
                    else:
                        raise RuntimeError
                elif preprocess == 'color_diff_121_abs_caffe':
                    if color_mode == 'RGB':
                        crop = crop - np.array([[[42.57, 44.33, 41.72, 43.35, 41.97, 43.85]]], dtype=np.float32)
                    else:
                        raise RuntimeError
                else:
                    raise RuntimeError
                crop = crop[np.newaxis, ...]
                batch_feature = np.concatenate((batch_feature, crop), 0)
                batch_label = np.concatenate((batch_label, label), 0)
        yield batch_feature, batch_label
        file_remain_num = file_remain_num - batch_size
        if file_remain_num < 1:
            file_remain_num = file_num
