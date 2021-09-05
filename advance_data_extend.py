import os
import tensorflow as tf
import numpy as np
import cv2
import random
from scipy import signal


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError("The input should be numpy ndarray. Instaed got {}".format(ndarray.dtype))


file_num = 1281167
thread_max = 32
split_max = 200

file_type = '.JPEG'
target_directory = 'E:/Dataset/Imagenet/ILSVRC2012_img_train/img/'
result_directory = 'E:/Dataset/Imagenet/ILSVRC2012_img_train/tfrecord/color_diff_121/'
result_tf_file = 'imagenet_train'
verbose = False
image_count = 0
class_count = 0
split_count = 0
walk_generator = os.walk(target_directory)
root, dir, _ = next(walk_generator)
class_num = len(dir)
# Generate tfrecord writer

file_path_list = []
for d in dir:
    # Generate tfrecord writer
    # writer = tf.io.TFRecordWriter(result_directory + result_tf_file + '.tfrecord-' + str(class_count).zfill(5) + '-of-' + str(class_num).zfill(5))
    print(d, class_count)
    walk_generator2 = os.walk(root + d)
    flies_root, _, files = next(walk_generator2)
    for file in files:
        if file.endswith(file_type):
            if not os.path.splitext(file)[0].endswith("_extend"):
                file_path_list.append({'path': os.path.join(flies_root, file), 'label': class_count})
    # writer.close()
    class_count = class_count + 1

writer = tf.io.TFRecordWriter(result_directory + result_tf_file + '.tfrecord-' + str(split_count).zfill(5))
random.seed(486)
shuffled_file_path_list = []
while len(file_path_list) > 0:
    rand_index = random.randrange(0, len(file_path_list), 1)
    shuffled_file_path_list.append(file_path_list[rand_index])
    del file_path_list[rand_index]
for instance in shuffled_file_path_list:
    image = cv2.imread(instance['path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # cv2.imshow("image", image)

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

    # custom edge
    # filter id 121
    horizontal_filter = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype="int")
    vertical_filter = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype="int")
    result = np.empty(shape=image.shape[:-1] + (0,))
    for RGB_index in range(3):
        horizontal_result = signal.convolve2d(image[..., RGB_index], horizontal_filter, boundary='symm',
                                              mode='same')
        vertical_result = signal.convolve2d(image[..., RGB_index], horizontal_filter, boundary='symm',
                                            mode='same')
        result = np.concatenate(
            (result, horizontal_result[..., np.newaxis], vertical_result[..., np.newaxis]), axis=-1)

    # numpy to tfrecord
    np_feature = result.reshape(-1)
    np_feature = np.array(np_feature, dtype=np.float32)
    np_label = np.zeros(class_num, dtype=np.int64)
    np_label[instance['label']] = 1

    dtype_feature_x = _dtype_feature(np_feature)
    dtype_feature_y = _dtype_feature(np_label)

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    d_feature = {'feature': dtype_feature_x(np_feature), 'label': dtype_feature_y(np_label)}
    features = tf.train.Features(feature=d_feature)
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
    image_count = image_count + 1
    if image_count % split_max == 0:
        writer.close()
        split_count = split_count + 1
        print(split_count)
        writer = tf.io.TFRecordWriter(
            result_directory + result_tf_file + '.tfrecord-' + str(split_count).zfill(5))

writer.close()
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

a = 0
