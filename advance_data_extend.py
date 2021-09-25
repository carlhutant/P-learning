import os
import tensorflow as tf
import numpy as np
import cv2
import random
import threading
import multiprocessing
from scipy import signal

thread_max = 48
split_max = 100
file_type = '.JPEG'
target_directory = 'E:/Dataset/imagenet/img/train/'
result_directory = 'E:/Dataset/imagenet/tfrecord/none/'
result_tf_file = 'train'
verbose = False
class_num = 1000
# save_file_type: origin, tfrecord
save_file_type = 'origin'
process_type = 'rgb_sw_021'


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    else:
        raise ValueError("The input should be numpy ndarray. Instaed got {}".format(ndarray.dtype))


def color_diff_121(array):
    # filter id 121
    horizontal_filter = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype="int")
    vertical_filter = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype="int")
    feature = np.empty(shape=array.shape[:-1] + (0,))
    for RGB_index in range(3):
        horizontal_result = signal.convolve2d(array[..., RGB_index], horizontal_filter, boundary='symm',
                                              mode='same')
        vertical_result = signal.convolve2d(array[..., RGB_index], vertical_filter, boundary='symm',
                                            mode='same')
        feature = np.concatenate(
            (feature, horizontal_result[..., np.newaxis], vertical_result[..., np.newaxis]), axis=-1)
    return feature


def np_instance_to_example(np_feature, np_label):
    # numpy to tfrecord
    np_feature = np_feature.reshape(-1)
    np_feature = np.array(np_feature, dtype=np.float32)
    np_label = np_label.reshape(-1)
    np_label = np.array(np_label, dtype=np.float32)

    dtype_feature_x = _dtype_feature(np_feature)
    dtype_feature_y = _dtype_feature(np_label)

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    d_feature = {'feature': dtype_feature_x(np_feature), 'label': dtype_feature_y(np_label)}
    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    return serialized


def file_load(instance_in):
    image = cv2.imread(instance_in['path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
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


def thread_func(id, instance_list):
    if save_file_type == 'tfrecord':
        split_count = 0
        image_count = 0
        writer = tf.io.TFRecordWriter(result_directory + result_tf_file + '.tfrecord-' + str(split_count*thread_max+id).zfill(5))
        for instance in instance_list:
            image, label = file_load(instance)
            feature = feature_processing(image)
            serialized_example = np_instance_to_example(feature, label)
            writer.write(serialized_example)
            image_count = image_count + 1
            if id == 0:
                print(str(image_count)+'/'+str(len(instance_list)))
            if image_count % split_max == 0:
                writer.close()
                split_count = split_count + 1
                # print(split_count)
                writer = tf.io.TFRecordWriter(
                    result_directory + result_tf_file + '.tfrecord-' + str(split_count*thread_max+id).zfill(5))
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
    walk_generator2 = os.walk('D:/Program/Python/P_learning/data/AWA2/test_draw/')  #
    root, directory, _ = next(walk_generator)
    _, directory2, _ = next(walk_generator2)
    class_num = len(directory)
    instance_list = []
    for i in range(len(directory)):
        if directory[i] != directory2[i]:
            a = 0
    for d in directory:
        print(d, class_count)
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        for file in files:
            if file.endswith(target_file_type):
                if not os.path.splitext(file)[0].endswith("_extend"):
                    instance_list.append({'path': os.path.join(flies_root, file), 'label': class_count})
        class_count = class_count + 1
    file_num = len(instance_list)

    random.seed(486)
    random.shuffle(instance_list)
    if thread_max == 0:
        no_thread_func(instance_list)
    else:
        threads = []
        for i in range(thread_max):
            threads.append(multiprocessing.Process(target=thread_func, args=(i, instance_list[i::thread_max])))
            # threads.append(threading.Thread(target=thread_func, args=(i, instance_list[i::thread_max])))
            threads[i].start()
        for i in range(thread_max):
            threads[i].join()
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
