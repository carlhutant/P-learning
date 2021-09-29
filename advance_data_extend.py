import os
import tensorflow as tf
import numpy as np
import cv2
import random
import multiprocessing
from scipy import signal

# Import data
# change the dataset here###
dataset = 'AWA2'
data_usage = 'train'
if dataset == 'AWA2':
    file_type = '.jpg'
    class_num = 40
elif dataset == 'imagenet':
    file_type = '.JPEG'
    class_num = 1000

process_max = 48
split_max = 15
target_directory = '/home/ai2020/ne6091069/Dataset/{}/img/none/{}/'.format(dataset, data_usage)
result_directory = '/home/ai2020/ne6091069/Dataset/{}/tfrecord/none/'.format(dataset)
result_tf_file = data_usage
verbose = False
# save_file_type: origin, tfrecord
save_file_type = 'tfrecord'
process_type = 'none'


# def color_diff_121(array):
#     # filter id 121
#     horizontal_filter = np.array([[1, 0, -1],
#                                   [2, 0, -2],
#                                   [1, 0, -1]], dtype="int")
#     vertical_filter = np.array([[1, 2, 1],
#                                 [0, 0, 0],
#                                 [-1, -2, -1]], dtype="int")
#     feature = np.empty(shape=array.shape[:-1] + (0,))
#     for RGB_index in range(3):
#         horizontal_result = signal.convolve2d(array[..., RGB_index], horizontal_filter, boundary='symm',
#                                               mode='same')
#         vertical_result = signal.convolve2d(array[..., RGB_index], vertical_filter, boundary='symm',
#                                             mode='same')
#         feature = np.concatenate(
#             (feature, horizontal_result[..., np.newaxis], vertical_result[..., np.newaxis]), axis=-1)
#     return feature


def np_instance_to_tf_example(np_shape, np_feature, np_label):
    # condirm data format
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

    # tf.train.Feature contain multiple tf.train.feature by mapping name and feature
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


def file_load(instance_in):
    image = cv2.imread(instance_in['path'])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # cv2.imshow("image", image)

    label = np.zeros(class_num, dtype=np.float32)
    label[instance_in['label']] = 1

    return image, label


def feature_processing(image):
    # # custom edge
    # # filter id 121
    # feature = color_diff_121(image)
    feature = image[..., [0, 2, 1]]
    return feature


def multiprocess_func(id, instance_list):
    if save_file_type == 'tfrecord':
        split_count = 0
        image_count = 0
        writer = tf.io.TFRecordWriter(
            result_directory + result_tf_file + '.tfrecord-' + str(split_count * process_max + id).zfill(5))
        for instance in instance_list:
            image, label = file_load(instance)
            shape = np.array(image.shape, dtype=np.int64)
            if process_type == 'none':
                feature = feature_processing(image)
                serialized_example = np_instance_to_tf_example(shape, feature, label)
            else:
                serialized_example = np_instance_to_tf_example(shape, image, label)
            writer.write(serialized_example)
            image_count = image_count + 1
            if id == 0:
                print(str(image_count) + '/' + str(len(instance_list)))
            if image_count % split_max == 0:
                writer.close()
                split_count = split_count + 1
                # print(split_count)
                writer = tf.io.TFRecordWriter(
                    result_directory + result_tf_file + '.tfrecord-' + str(split_count * process_max + id).zfill(5))
        writer.close()
    elif save_file_type == 'origin':
        for instance in instance_list:
            image, _ = file_load(instance)
            feature = feature_processing(image)
            split_file_name = os.path.splitext(instance['path'])
            save_file_name = split_file_name[0] + '_' + process_type + split_file_name[1]
            cv2.imwrite(save_file_name, feature)


def no_thread_func(instance_list):
    split_count = 0
    image_count = 0
    writer = tf.io.TFRecordWriter(result_directory + result_tf_file + '.tfrecord-' + str(split_count).zfill(5))
    for instance in instance_list:
        serialized_example = file_processing(instance)
        writer.write(serialized_example)
        image_count = image_count + 1
        if image_count % split_max == 0:
            writer.close()
            split_count = split_count + 1
            # print(split_count)
            writer = tf.io.TFRecordWriter(
                result_directory + result_tf_file + '.tfrecord-' + str(split_count).zfill(5))
    writer.close()


if __name__ == "__main__":
    class_count = 0
    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    class_num = len(directory)
    instance_list = []
    for d in directory:
        print(d, class_count)
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        for file in files:
            if file.endswith(file_type):
                if not os.path.splitext(file)[0].endswith("_extend"):
                    instance_list.append({'path': os.path.join(flies_root, file), 'label': class_count})
        class_count = class_count + 1
    file_num = len(instance_list)

    random.seed(486)
    random.shuffle(instance_list)
    if process_max == 0:
        no_thread_func(instance_list)
    else:
        processes = []
        for i in range(process_max):
            processes.append(multiprocessing.Process(target=multiprocess_func, args=(i, instance_list[i::process_max])))
            processes[i].start()
        for i in range(process_max):
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
