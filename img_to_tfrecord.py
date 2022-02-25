import os
import tensorflow as tf
import numpy as np
import cv2
import random
import multiprocessing
from pathlib import Path
from configure import *

# color_diff_121:
#   0.00051614,  0.01895578, -0.00204118, -0.03210322, -0.00153458, -0.12204278
# color_diff_121_abs RGB:
#   42.57167483915786, 44.32660178095038, 41.716144938386016, 43.35167134089522, 41.97310964205989, 43.8454831598209


multiprocess = False
process_num = 1
split_instance_num = 353
dataset = 'AWA2'
origin_datatype = 'img'
result_datatype = 'tfrecord'  # img, tfrecord, npy
# color_diff_121, color_diff_121_abs_3ch, color_diff_121_abs, none, color_sw_GBR, color_diff_121_abs_3ch
origin_data_advance = 'color_diff_121_abs_3ch'
result_data_advance = 'color_diff_121_abs_3ch'
data_usage = 'train'  # data usage: train, val, test

target_directory = Path('{}/{}/{}/{}/{}/'.format(dataset_dir, dataset, origin_datatype, origin_data_advance, data_usage))
result_directory = Path('{}/{}/{}/{}/'.format(dataset_dir, dataset, result_datatype, result_data_advance))
if result_datatype != 'tfrecord':
    result_directory = result_directory.joinpath(data_usage)

if dataset == 'AWA2':
    file_type = '.jpg'
    class_num = 40
elif dataset == 'imagenet':
    file_type = '.JPEG'
    class_num = 1000
elif dataset == 'AWA1':
    file_type = '.jpg'
    class_num = 3
else:
    file_type = 'none'
    class_num = -1
    raise RuntimeError


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    elif isinstance(value, bytes):
        pass
    else:
        raise RuntimeError
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        raise RuntimeError


def _int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        raise RuntimeError


def file_load_to_feature(instance):
    f = open(str(instance['path']), 'rb')
    img_encoded_bytes = f.read()
    one_hot_label = [0.0]*class_num
    one_hot_label[instance['label']] = 1.0

    feature = _bytes_feature(img_encoded_bytes)
    label = _float_feature(one_hot_label)

    return feature, label


def feature_to_example(feature, label):
    # tf.train.Features contain multiple tf.train.feature by mapping name with feature
    features = tf.train.Features(
        feature={
            'feature': feature,
            'label': label
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def process_func(process_id, partial_instance_list):
    split_count = 0
    image_count = 0
    writer = tf.io.TFRecordWriter(str(
        result_directory.joinpath(data_usage + '.tfrecord-' + str(split_count * process_num + process_id).zfill(5))))
    finished_count = 0
    for instance in partial_instance_list:
        if process_id == 0:
            finished_count = finished_count + 1
            # print(instance['path'])
            print('{}/{}'.format(finished_count, len(partial_instance_list)))
        feature, label = file_load_to_feature(instance)
        serialized_example = feature_to_example(feature, label)
        writer.write(serialized_example)
        image_count = image_count + 1
        # if process_id == 0:
        #     print(str(image_count) + '/' + str(len(partial_instance_list)))
        if image_count % split_instance_num == 0:
            writer.close()
            split_count = split_count + 1
            # print(split_count)
            writer = tf.io.TFRecordWriter(
                str(result_directory.joinpath(
                    data_usage + '.tfrecord-' + str(split_count * process_num + process_id).zfill(5))))
    if result_datatype == 'tfrecord':
        writer.close()
    return


if __name__ == "__main__":
    class_count = 0
    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    class_num = len(directory)
    instance_list = []
    for d in directory:
        print(d, class_count)
        walk_generator2 = os.walk(Path(root).joinpath(d))
        flies_root, _, files = next(walk_generator2)
        for file in files:
            if file.endswith(file_type):
                instance_list.append({'path': Path(flies_root).joinpath(file), 'label': class_count})
        class_count = class_count + 1
    file_num = len(instance_list)

    random.seed(486)
    random.shuffle(instance_list)

    if not result_datatype == 'tfrecord':
        if not Path.is_dir(Path(result_directory).parent.parent):
            Path.mkdir(Path(result_directory).parent.parent)
    if not Path.is_dir(Path(result_directory).parent):
        Path.mkdir(Path(result_directory).parent)
    if not Path.is_dir(Path(result_directory)):
        Path.mkdir(Path(result_directory))
    if not result_datatype == 'tfrecord':
        for d in directory:
            if not Path.is_dir(Path(result_directory).joinpath(d)):
                Path.mkdir(Path(result_directory).joinpath(d))

    if not multiprocess:
        process_num = 1
        process_func(0, instance_list)
    else:
        processes = []
        for i in range(process_num):
            processes.append(multiprocessing.Process(target=process_func, args=(i, instance_list[i::process_num])))
            processes[i].start()
        for i in range(process_num):
            processes[i].join()
        print('done')
