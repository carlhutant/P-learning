import os
import cv2
import math
import random
import numpy as np
import multiprocessing
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from pathlib import Path
from configure import *


def load_generator(target_directory, shuffle, shuffle_every_epoch):
    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    instance_list = []
    class_count = 0
    directory.sort()
    for d in directory:
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        files.sort()
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
            elif datatype == 'npy':
                img = np.load(instance['path'])
            else:
                raise RuntimeError
            img = np.array(img, dtype=np.float32)

            label = np.zeros(class_num, dtype=np.float32)
            label[instance['label']] = 1
            # show dolphin sw021 img
            # if instance['label'] == 11:
            #     print(instance['path'])
            #     img = np.array(img[..., [0, 2, 1]], dtype=np.uint8)
            #     cv2.imshow('123', img)
            #     cv2.waitKey()
            yield img, label
        if shuffle_every_epoch:
            random.shuffle(instance_list)


def domains_load_generator(target_directory, target2_directory, shuffle, shuffle_every_epoch):
    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    instance_list = []
    class_count = 0
    directory.sort()
    for d in directory:
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        files.sort()
        for file in files:
            instance_list.append({'path': Path(flies_root).joinpath(file),
                                  'path2': Path(target2_directory).joinpath(d).joinpath(file),
                                  'label': class_count})
        class_count += 1
    file_num = len(instance_list)
    if shuffle or shuffle_every_epoch:
        random.shuffle(instance_list)

    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    class_count2 = 0
    file_num2 = 0
    for d in directory:
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        class_count2 += 1
        file_num2 += len(files)

    yield file_num, class_count, file_num2, class_count2
    while True:
        for i in range(file_num):
            instance = instance_list[i]
            if datatype == 'img':
                feature = cv2.imread(str(instance['path']))
            elif datatype == 'npy':
                feature = np.load(str(instance['path']))
            else:
                raise RuntimeError

            if datatype2 == 'img':
                feature2 = cv2.imread(str(instance['path2']))
            elif datatype2 == 'npy':
                feature2 = np.load(str(instance['path2']))
            else:
                raise RuntimeError

            # cv2.imshow('1', feature)
            # cv2.imshow('2', feature2)
            # cv2.waitKey()

            feature = np.array(feature, dtype=np.float32)
            feature2 = np.array(feature2, dtype=np.float32)

            label = np.zeros(class_num, dtype=np.float32)
            label[instance['label']] = 1
            yield feature, feature2, label
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
    # # testing load_generator speed
    # count = 0
    # count2 = 0
    # while True:
    #     a = next(img_gen)
    #     count += 1
    #     if count % batch_size == 0:
    #         count2 += 1
    #         print(count2)
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
                    crop = crop - np.array([[pixel_mean]], dtype=np.float32)
                elif preprocess == 'none':
                    pass
                else:
                    raise RuntimeError
                if color_mode == 'RGB':
                    if data_advance == 'none':
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    elif data_advance.startswith('color_diff'):
                        crop = crop[..., [4, 5, 2, 3, 0, 1]]
                    else:
                        raise RuntimeError
                if color_mode == 'GBR':
                    if data_advance == 'none':
                        crop = crop[..., [1, 0, 2]]
                    else:
                        raise RuntimeError
                crop = crop[np.newaxis, ...]
                batch_feature = np.concatenate((batch_feature, crop), 0)
                batch_label = np.concatenate((batch_label, label), 0)
        yield batch_feature, batch_label
        file_remain_num = file_remain_num - batch_size
        if file_remain_num < 1:
            file_remain_num = file_num


def domains_feature_generator(target_directory, target2_directory, model, model2, batch_size, final_batch_opt,
                              crop_type, crop_h, crop_w, resize_short_edge_max, resize_short_edge_min, horizontal_flip,
                              shuffle, shuffle_every_epoch):
    img_gen = domains_load_generator(target_directory, target2_directory, shuffle, shuffle_every_epoch)
    file_num, dir_num, file_num2, dir_num2 = next(img_gen)
    print("Found {} images belonging to {} classes.".format(file_num, dir_num))
    if file_num != train_cardinality and file_num != val_cardinality:
        raise RuntimeError
    if dir_num != class_num:
        raise RuntimeError
    if file_num2 != train_cardinality and file_num2 != val_cardinality:
        raise RuntimeError
    if dir_num2 != class_num:
        raise RuntimeError
    yield

    random.seed(random_seed)
    file_remain_num = file_num
    while True:
        batch_feature = np.empty([0, crop_h, crop_w, channel], dtype=np.float32)
        batch_feature2 = np.empty([0, crop_h, crop_w, channel2], dtype=np.float32)
        batch_label = np.empty([0, class_num], dtype=np.float32)
        if final_batch_opt == 'complete':
            batch_data_num = min(batch_size, file_remain_num)
        elif final_batch_opt == 'full':
            batch_data_num = batch_size
        else:
            print('final_batch_opt error.')
            raise RuntimeError
        for i in range(batch_data_num):
            feature, feature2, label = next(img_gen)
            concatenate_feature = np.concatenate((feature, feature2), axis=-1)
            label = label[np.newaxis, ...]
            if horizontal_flip:
                if random.randint(0, 1):
                    concatenate_feature = concatenate_feature[:, ::-1, :]
            height, width, _ = concatenate_feature.shape
            if crop_type == 'none':
                new_height = crop_h
                new_width = crop_w
            elif height < width:
                new_height = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_width = round(width * new_height / height)
            else:
                new_width = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_height = round(height * new_width / width)
            concatenate_feature = cv2.resize(concatenate_feature, (new_width, new_height))

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
            concatenate_crop_list = []
            for y0, x0 in zip(y0_list, x0_list):
                crop = concatenate_feature[y0:y0 + crop_h, x0:x0 + crop_w, :]
                concatenate_crop_list.append(crop)
                if crop_type == 'ten_crop':
                    concatenate_crop_list.append(crop[:, ::-1, :])

            crop_list = []
            crop_list2 = []
            for concatenate_crop in concatenate_crop_list:
                crop_list.append(concatenate_crop[..., 0:channel])
                crop_list2.append(concatenate_crop[..., channel:])
            for crop in crop_list:
                if preprocess == 'caffe':
                    crop = crop - np.array([[pixel_mean]], dtype=np.float32)
                elif preprocess == 'none':
                    pass
                else:
                    raise RuntimeError
                if color_mode == 'RGB':
                    if data_advance == 'none' or data_advance.endswith('3ch'):
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    elif data_advance.startswith('color_diff'):
                        crop = crop[..., [4, 5, 2, 3, 0, 1]]
                    else:
                        raise RuntimeError
                if color_mode == 'GBR':
                    if data_advance == 'none':
                        crop = crop[..., [1, 0, 2]]
                    else:
                        raise RuntimeError
                crop = crop[np.newaxis, ...]
                batch_feature = np.concatenate((batch_feature, crop), 0)
                batch_label = np.concatenate((batch_label, label), 0)

            for crop in crop_list2:
                if preprocess2 == 'caffe':
                    crop = crop - np.array([[pixel_mean2]], dtype=np.float32)
                elif preprocess2 == 'none':
                    pass
                else:
                    raise RuntimeError
                if color_mode2 == 'RGB':
                    if data_advance2 == 'none' or data_advance2.endswith('3ch'):
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    elif data_advance2.startswith('color_diff'):
                        crop = crop[..., [4, 5, 2, 3, 0, 1]]
                    else:
                        raise RuntimeError
                if color_mode2 == 'GBR':
                    if data_advance2 == 'none':
                        crop = crop[..., [1, 0, 2]]
                    else:
                        raise RuntimeError
                crop = crop[np.newaxis, ...]
                batch_feature2 = np.concatenate((batch_feature2, crop), 0)
        batch_feature2048 = model.predict(batch_feature)
        batch_feature2048_2 = model2.predict(batch_feature2)
        yield np.concatenate((batch_feature2048, batch_feature2048_2), axis=1), batch_label
        file_remain_num = file_remain_num - batch_size
        if file_remain_num < 1:
            file_remain_num = file_num


def parallel_data_loader(id_, buffer, config):
    walk_generator = os.walk(config['target_directory'])
    root, directory, _ = next(walk_generator)
    instance_list = []
    class_count = 0
    directory.sort()
    for d in directory:
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        files.sort()
        for file in files:
            instance_list.append({'path': os.path.join(flies_root, file), 'label': class_count})
        class_count = class_count + 1
    file_num = len(instance_list)
    if file_num != train_cardinality and file_num != val_cardinality:
        raise RuntimeError
    if class_count != class_num:
        raise RuntimeError

    if config['shuffle'] or config['shuffle_every_epoch']:
        random.shuffle(instance_list)
    if id_ == 0:
        print("Found {} images belonging to {} classes.".format(file_num, class_count))

    while True:
        for instance in instance_list[:id_:config['process_num']]:
            if datatype == 'img':
                img = cv2.imread(instance['path'])
            elif datatype == 'npy':
                img = np.load(instance['path'])
            else:
                raise RuntimeError
            img = np.array(img, dtype=np.float32)

            label = np.zeros(class_num, dtype=np.float32)
            label[instance['label']] = 1
            # show dolphin sw021 img
            # if instance['label'] == 11:
            #     print(instance['path'])
            #     img = np.array(img[..., [0, 2, 1]], dtype=np.uint8)
            #     cv2.imshow('123', img)
            #     cv2.waitKey()
            while len(buffer) > 10:
                pass
            buffer.append((img, label))
            print('buffer {}: {}'.format(id_, len(buffer)))
        if config['shuffle_every_epoch']:
            random.shuffle(instance_list)


def parallel_crop_generator(input_buffer, output_buffer, config):
    random.seed(random_seed)
    while True:
        while len(input_buffer) < 1:
            pass

        img, label = input_buffer[0]
        del input_buffer[0]
        label = label[np.newaxis, ...]
        if config['horizontal_flip']:
            if random.randint(0, 1):
                img = img[:, ::-1, :]
        height, width, _ = img.shape
        if config['crop_type'] == 'none':
            new_height = ['crop_h']
            new_width = ['crop_w']
        elif height < width:
            new_height = random.randint(config['resize_short_edge_min'], config['resize_short_edge_max'])
            new_width = round(width * new_height / height)
        else:
            new_width = random.randint(config['resize_short_edge_min'], config['resize_short_edge_max'])
            new_height = round(height * new_width / width)
        img = cv2.resize(img, (new_width, new_height))

        y0_list = []
        x0_list = []
        if config['crop_type'] == 'none':
            y0_list.append(0)
            x0_list.append(0)
        elif config['crop_type'] == 'center':
            center_y = math.ceil(new_height / 2)
            center_x = math.ceil(new_width / 2)
            y0_list.append(center_y - math.ceil(config['crop_h'] / 2) + 1)
            x0_list.append(center_x - math.ceil(config['crop_w'] / 2) + 1)
        elif config['crop_type'] == 'random':  # 先不處理 crop 大於原圖的情況
            y0_list.append(random.randint(0, new_height - config['crop_h']))
            x0_list.append(random.randint(0, new_width - config['crop_w']))
        elif config['crop_type'] == 'ten_crop':
            center_y = math.ceil(new_height / 2)
            center_x = math.ceil(new_width / 2)
            y0_list.append(center_y - math.ceil(config['crop_h'] / 2) + 1)
            x0_list.append(center_x - math.ceil(config['crop_w'] / 2) + 1)
            y0_list.extend([0, 0, new_height - config['crop_h'], new_height - config['crop_h']])
            x0_list.extend([0, new_width - config['crop_h'], 0, new_width - config['crop_h']])
        else:
            raise RuntimeError
        crop_list = []
        for y0, x0 in zip(y0_list, x0_list):
            crop = img[y0:y0 + config['crop_h'], x0:x0 + config['crop_w'], :]
            crop_list.append(crop)
            if config['crop_type'] == 'ten_crop':
                crop_list.append(crop[:, ::-1, :])
        total_crop = np.empty([0, config['crop_h'], config['crop_w'], channel], dtype=np.float32)
        total_label = np.empty([0, class_num], dtype=np.float32)
        for crop in crop_list:
            if preprocess == 'caffe':
                crop = crop - np.array([[pixel_mean]], dtype=np.float32)
            elif preprocess == 'none':
                pass
            else:
                raise RuntimeError
            if color_mode == 'RGB':
                if data_advance == 'none':
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                elif data_advance.startswith('color_diff'):
                    crop = crop[..., [4, 5, 2, 3, 0, 1]]
                else:
                    raise RuntimeError
            if color_mode == 'GBR':
                if data_advance == 'none':
                    crop = crop[..., [1, 0, 2]]
                else:
                    raise RuntimeError
            crop = crop[np.newaxis, ...]
            total_crop = np.concatenate((total_crop, crop), 0)
            total_label = np.concatenate((total_label, label), 0)
        while len(output_buffer) > 3 * max(train_batch_size, val_batch_size):
            pass
        output_buffer.append((total_crop, total_label))


def parallel_batch_generator(target_directory, batch_size, final_batch_opt, crop_type, crop_h, crop_w,
                             resize_short_edge_max, resize_short_edge_min, horizontal_flip, shuffle, shuffle_every_epoch
                             ):
    process_num = 4
    manager = multiprocessing.Manager()
    load_buffer_list = []
    crop_buffer_list = []
    for process_No in range(process_num):
        load_buffer = manager.list()
        crop_buffer = manager.list()
        load_buffer_list.append(load_buffer)
        crop_buffer_list.append(crop_buffer)

    config = manager.dict()
    config['process_num'] = process_num
    config['target_directory'] = target_directory
    config['batch_size'] = batch_size
    config['final_batch_opt'] = final_batch_opt
    config['crop_type'] = crop_type
    config['crop_h'] = crop_h
    config['crop_w'] = crop_w
    config['resize_short_edge_max'] = resize_short_edge_max
    config['resize_short_edge_min'] = resize_short_edge_min
    config['horizontal_flip'] = horizontal_flip
    config['shuffle'] = shuffle
    config['shuffle_every_epoch'] = shuffle_every_epoch
    config['file_num'] = -1
    config['dir_num'] = -1

    pdl_list = []
    pcl_list = []
    for process_No in range(process_num):
        pdl = multiprocessing.Process(target=parallel_data_loader,
                                      args=(process_No, load_buffer_list[process_No], config),
                                      name='pdl-{}'.format(process_No))
        pdl_list.append(pdl)
        pcl = multiprocessing.Process(target=parallel_crop_generator,
                                      args=(load_buffer_list[process_No], crop_buffer_list[process_No], config),
                                      name='pcl-{}'.format(process_No))
        pcl_list.append(pcl)
    for process_No in range(process_num):
        pdl_list[process_No].start()
        pcl_list[process_No].start()
    while config['file_num'] == -1:
        pass

    while True:
        batch_feature = np.empty([0, crop_h, crop_w, channel], dtype=np.float32)
        batch_label = np.empty([0, class_num], dtype=np.float32)
        file_remain_num = config['file_num']
        if final_batch_opt == 'complete':
            batch_data_num = min(batch_size, file_remain_num)
        elif final_batch_opt == 'full':
            batch_data_num = batch_size
        else:
            print('final_batch_opt error.')
            raise RuntimeError
        batch_data_count = 0
        process_No = 0
        while batch_data_count < batch_data_num:
            while len(crop_buffer_list[process_No]) < 1:
                pass
            batch_feature = np.concatenate((batch_feature, crop_buffer_list[process_No][0][0]), axis=0)
            batch_label = np.concatenate((batch_label, crop_buffer_list[process_No][0][1]), axis=0)
            del crop_buffer_list[process_No][0]
            batch_data_count += 1
            process_No = (process_No + 1) % process_num

        signal = yield batch_feature, batch_label
        if signal is not None:

            for process_No in range(process_num):
                pdl_list[process_No].terminate()
                pcl_list[process_No].terminate()
                pdl_list[process_No].join()
                pcl_list[process_No].join()
            manager.shutdown()
            while True:
                yield
        file_remain_num = file_remain_num - batch_size
        if file_remain_num < 1:
            file_remain_num = config['file_num']
