import configure
import os
import tensorflow as tf
import numpy as np
import cv2
import random
import multiprocessing
from pathlib import Path
from scipy import signal


# color_diff_121:
#   0.00051614,  0.01895578, -0.00204118, -0.03210322, -0.00153458, -0.12204278
# color_diff_121_abs:
#   42.57167483915786, 44.32660178095038, 41.716144938386016, 43.35167134089522, 41.97310964205989, 43.8454831598209
multiprocess = False
process_num = 1
split_instance_num = 10
dataset = 'AWA2'
result_datatype = 'tfrecord'    # result_datatype: img, tfrecord, npy
data_advance = 'none'   # data data_advance: color_diff_121, color_diff_121_3ch, color_diff_121_abs, none
data_usage = 'train'  # data usage: train, val, test

dataset_dir = configure.dataset_dir
target_directory = Path('{}/{}/img/none/{}/'.format(dataset_dir, dataset, data_usage))
result_directory = Path('{}/{}/{}/{}/'.format(dataset_dir, dataset, result_datatype, data_advance))
if result_datatype != 'tfrecord':
    result_directory = result_directory.joinpath(data_usage)

if dataset == 'AWA2':
    file_type = '.jpg'
    class_num = 40
elif dataset == 'imagenet':
    file_type = '.JPEG'
    class_num = 1000
else:
    file_type = 'none'
    class_num = -1
    raise RuntimeError


def color_diff_121(array):
    horizontal_filter = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype="int")
    vertical_filter = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype="int")
    feature = np.empty(shape=array.shape[:-1] + (0,))
    for BGR_index in range(3):
        horizontal_result = signal.convolve2d(array[..., BGR_index], horizontal_filter, boundary='symm',
                                              mode='same')
        vertical_result = signal.convolve2d(array[..., BGR_index], vertical_filter, boundary='symm',
                                            mode='same')
        feature = np.concatenate(
            (feature, horizontal_result[..., np.newaxis], vertical_result[..., np.newaxis]), axis=-1)
    return feature


def color_diff_121_abs(array):
    feature = color_diff_121(array)
    return np.absolute(feature)


def np_instance_to_tf_example(np_shape, np_feature, np_label):
    # confirm data format
    np_shape = np_shape.reshape(-1)
    np_shape = np.array(np_shape, dtype=np.int64)
    np_feature = np_feature.reshape(-1)
    np_feature = np.array(np_feature, dtype=np.float32)
    np_label = np_label.reshape(-1)
    np_label = np.array(np_label, dtype=np.float32)

    # ndarray to tf.train.Feature
    tf_feature_shape = tf.train.Feature(int64_list=tf.train.Int64List(value=np_shape))
    tf_feature_feature = tf.train.Feature(float_list=tf.train.FloatList(value=np_feature))
    tf_feature_label = tf.train.Feature(float_list=tf.train.FloatList(value=np_label))

    # tf.train.Features contain multiple tf.train.feature by mapping name and feature
    tf_features = tf.train.Features(
        feature={
            'shape:': tf_feature_shape,
            'feature': tf_feature_feature,
            'label': tf_feature_label
        }
    )
    example = tf.train.Example(features=tf_features)
    serialized_example = example.SerializeToString()
    return serialized_example


def file_load(instance):
    image = cv2.imread(str(instance['path']))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    label = np.zeros(class_num, dtype=np.float32)
    label[instance['label']] = 1
    return image, label


def feature_processing(image):
    feature = color_diff_121_abs(image)
    # feature = image[..., [0, 2, 1]]
    # feature = image
    return feature


def process_func(process_id, partial_instance_list):
    if result_datatype == 'tfrecord':
        split_count = 0
        image_count = 0
        writer = tf.io.TFRecordWriter(
            str(result_directory.joinpath(
                data_usage + '.tfrecord-' + str(split_count * process_num + process_id).zfill(5))))

    for instance in partial_instance_list:
        print(instance['path'])
        image, label = file_load(instance)
        shape = np.array(image.shape, dtype=np.int64)
        feature = feature_processing(image)

        if result_datatype == 'tfrecord':
            serialized_example = np_instance_to_tf_example(shape, feature, label)
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
        elif result_datatype == 'npy':
            np.save(str(Path(result_directory).joinpath(Path(instance['path']).parent.stem)
                        .joinpath(Path(instance['path']).stem)), feature)
        elif result_datatype == 'img':
            cv2.imwrite(str(Path(result_directory).joinpath(Path(instance['path']).parent.stem)
                        .joinpath(str(Path(instance['path']).name))), feature)
        else:
            raise RuntimeError
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
    if result_datatype == 'tfrecord':
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

    # # special edge
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    # # cv2.imshow("blurred", blurred)
    # # 兩個 threshold, 微分高於 upper 為邊界, 高於 lower 周圍有高於 upper 為邊界
    # canny = cv2.Canny(blurred, 20, 40)
    # result = np.ones(canny.shape, dtype="uint8")*255-canny
    # # cv2.imshow("canny", canny)
    # # cv2.waitKey(0)
    # result = result[:, :, np.newaxis]
    # result = np.array(result, dtype="uint8")
    # result_n = np.ones(result.shape, dtype="uint8") * 255 - result
    # one = np.ones(result.shape, dtype="uint8") * 255
    # zero = np.zeros(result.shape, dtype="uint8")
    # result = np.concatenate((result, result_n, result_n), axis=-1)
    # origin_file_name = os.path.splitext(os.path.join(rr, file))
    #
    # cv2.imwrite(origin_file_name[0] + origin_file_name[1], result)

    # # edge & negative edge
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    # # cv2.imshow("blurred", blurred)
    # # 兩個 threshold, 微分高於 upper 為邊界, 高於 lower 周圍有高於 upper 為邊界
    # canny = cv2.Canny(blurred, 20, 40)
    # # result = np.ones(canny.shape, dtype="uint8")*255-canny
    # # cv2.imshow("canny", canny)
    # # cv2.waitKey(0)
    # canny = canny[:, :, np.newaxis]
    # result = np.concatenate((canny, canny, canny), axis=-1)
    # origin_file_name = os.path.splitext(os.path.join(rr, file))
    #
    # cv2.imwrite(origin_file_name[0] + origin_file_name[1], canny)
    # # cv2.imwrite(origin_file_name[0] + '_edge_negative_extend' + origin_file_name[1], result)
    # image = cv2.imread(os.path.join(r, file))
    #     # cv2.imshow("image", image)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    #     # cv2.imshow("blurred", blurred)
    #     # 兩個 threshold, 微分高於 upper 為邊界, 高於 lower 周圍有高於 upper 為邊界
    #     canny = cv2.Canny(blurred, 20, 40)
    #     # cv2.imshow("canny", canny)
    #     # cv2.waitKey(0)
    #     canny = canny[:, :, np.newaxis]
    #     result = np.concatenate((canny, canny, canny), axis=-1)
    #     origin_file_name = os.path.splitext(os.path.join(r, file))
    #     cv2.imwrite(origin_file_name[0] + '_edge_extend' + origin_file_name[1], result)
    #
    #     image_split = np.split(image, 3, axis=-1)
    #     reorder_021 = np.concatenate((image_split[0], image_split[2], image_split[1]), -1)
    #     reorder_102 = np.concatenate((image_split[1], image_split[0], image_split[2]), -1)
    #     reorder_120 = np.concatenate((image_split[1], image_split[2], image_split[0]), -1)
    #     reorder_201 = np.concatenate((image_split[2], image_split[0], image_split[1]), -1)
    #     reorder_210 = np.concatenate((image_split[2], image_split[1], image_split[0]), -1)
    #     # cv2.imshow("reorder_012", reorder_012)
    #     # cv2.imshow("reorder_102", reorder_102)
    #     # cv2.imshow("reorder_120", reorder_120)
    #     # cv2.imshow("reorder_201", reorder_201)
    #     # cv2.imshow("reorder_210", reorder_210)
    #     negative_012 = (np.ones(image.shape, dtype="uint8") * 255) - image
    #     negative_021 = (np.ones(image.shape, dtype="uint8") * 255) - reorder_021
    #     negative_102 = (np.ones(image.shape, dtype="uint8") * 255) - reorder_102
    #     negative_120 = (np.ones(image.shape, dtype="uint8") * 255) - reorder_120
    #     negative_201 = (np.ones(image.shape, dtype="uint8") * 255) - reorder_201
    #     negative_210 = (np.ones(image.shape, dtype="uint8") * 255) - reorder_210
    #     # cv2.imshow("negative_012", negative_012)
    #     # cv2.imshow("negative_021", negative_021)
    #     # cv2.imshow("negative_102", negative_102)
    #     # cv2.imshow("negative_120", negative_120)
    #     # cv2.imshow("negative_201", negative_201)
    #     # cv2.imshow("negative_210", negative_210)
    #     # cv2.waitKey(0)
    #     origin_file_name = os.path.splitext(os.path.join(r, file))
    #     cv2.imwrite(origin_file_name[0] + '_reorder_021_extend' + origin_file_name[1], reorder_021)
    #     cv2.imwrite(origin_file_name[0] + '_reorder_102_extend' + origin_file_name[1], reorder_102)
    #     cv2.imwrite(origin_file_name[0] + '_reorder_120_extend' + origin_file_name[1], reorder_120)
    #     cv2.imwrite(origin_file_name[0] + '_reorder_201_extend' + origin_file_name[1], reorder_201)
    #     cv2.imwrite(origin_file_name[0] + '_reorder_210_extend' + origin_file_name[1], reorder_210)
    #     cv2.imwrite(origin_file_name[0] + '_negative_012_extend' + origin_file_name[1], negative_012)
    #     cv2.imwrite(origin_file_name[0] + '_negative_021_extend' + origin_file_name[1], negative_021)
    #     cv2.imwrite(origin_file_name[0] + '_negative_102_extend' + origin_file_name[1], negative_102)
    #     cv2.imwrite(origin_file_name[0] + '_negative_120_extend' + origin_file_name[1], negative_120)
    #     cv2.imwrite(origin_file_name[0] + '_negative_201_extend' + origin_file_name[1], negative_201)
    #     cv2.imwrite(origin_file_name[0] + '_negative_210_extend' + origin_file_name[1], negative_210)

    # # edge & negative edge
    # image = cv2.imread(os.path.join(r, file))
    # # cv2.imshow("image", image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    # # cv2.imshow("blurred", blurred)
    # # 兩個 threshold, 微分高於 upper 為邊界, 高於 lower 周圍有高於 upper 為邊界
    # canny = cv2.Canny(gray, 20, 40)
    # result = np.ones(canny.shape, dtype="uint8")*255-canny
    # # cv2.imshow("canny", canny)
    # # cv2.waitKey(0)
    # result = result[:, :, np.newaxis]
    # result = np.concatenate((result, result, result), axis=-1)
    # origin_file_name = os.path.splitext(os.path.join(r, file))
    #
    # # cv2.imwrite(origin_file_name[0] + '_edge_extend' + origin_file_name[1], canny)
    # # cv2.imwrite(origin_file_name[0] + '_edge_negative_extend' + origin_file_name[1], result)

    # # edge to negative edge
    # if os.path.splitext(file)[0].endswith("_edge_extend"):
    #     image = cv2.imread(os.path.join(r, file))
    #     result = np.ones(image.shape, dtype="uint8")*255-image
    #     result[..., 0] = np.ones(image[..., 0].shape, dtype="uint8")*255
    #     result[..., 1] = np.zeros(image[..., 1].shape, dtype="uint8")
    #     origin_file_name = os.path.splitext(os.path.join(r, file))
    #     cv2.imwrite(origin_file_name[0] + '_edge_negative_extend' + origin_file_name[1], result)
