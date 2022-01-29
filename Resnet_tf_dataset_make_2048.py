import tensorflow as tf
import numpy as np
import cv2
import gc

from tensorflow.keras.models import Model, load_model
from configure import *
from pathlib import Path


def tf_parse(raw_example):
    example = tf.io.parse_example(
        raw_example[tf.newaxis], {
            'shape': tf.io.VarLenFeature(dtype=tf.int64),
            'feature': tf.io.VarLenFeature(dtype=tf.float32),
            'label': tf.io.VarLenFeature(dtype=tf.float32)
        })
    shape = tf.sparse.to_dense(example['shape'])
    shape = tf.get_static_value(shape)
    feature = tf.reshape(example['feature'][0], shape)
    label = example['label'][0]
    return feature, label


def example_parse_decode(raw_example):
    example = tf.io.parse_example(
        raw_example, {
            'feature': tf.io.VarLenFeature(dtype=tf.string),
            'label': tf.io.FixedLenFeature(dtype=tf.float32, shape=(class_num,))
        })
    dense = tf.sparse.to_dense(example['feature'])[0]
    feature = tf.io.decode_jpeg(dense, channels=3)
    feature = tf.cast(feature, dtype=tf.float32)
    label = example['label']
    return feature, label


def random_crop(img, label, config):
    shape = tf.shape(img)
    shape = tf.slice(shape, [0], [2])
    shape_min = tf.reduce_min(shape)
    target_short_edge = tf.random.uniform(shape=[], minval=config['resize_short_edge_min'],
                                          maxval=config['resize_short_edge_max'] + 1, dtype=tf.int32)
    ratio = tf.divide(target_short_edge, shape_min)
    shape = tf.multiply(tf.cast(shape, dtype=tf.float64), ratio)
    shape = tf.cast(shape, dtype=tf.int32)
    img = tf.image.resize(images=img, size=shape, preserve_aspect_ratio=True)
    crop = tf.image.random_crop(value=img, size=(config['crop_h'], config['crop_w'], config['channel']))
    return crop, label


def random_plus2_crop(img, label, config):
    shape = tf.shape(img)
    shape = tf.slice(shape, [0], [2])
    shape_min = tf.reduce_min(shape)
    target_short_edge = tf.random.uniform(shape=[], minval=config['resize_short_edge_min'],
                                          maxval=config['resize_short_edge_max'] + 1, dtype=tf.int32)
    ratio = tf.divide(target_short_edge, shape_min)
    shape = tf.multiply(tf.cast(shape, dtype=tf.float64), ratio)
    shape = tf.cast(shape, dtype=tf.int32)
    img = tf.image.resize(images=img, size=shape, preserve_aspect_ratio=True)
    crop = tf.image.random_crop(value=img, size=(config['crop_h'] + 2, config['crop_w'] + 2, config['channel']))
    return crop, label


def color_diff_121(img):
    shape = tf.shape(img)
    img = tf.expand_dims(img, 0)
    filter_h = np.array([
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]],
        [[[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-2, 0, 0], [0, -2, 0], [0, 0, -2]]],
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]]
    ])
    filter_v = np.array([
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        [[[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [[-2, 0, 0], [0, -2, 0], [0, 0, -2]],
         [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]]
    ])
    filter_h = tf.constant(filter_h, dtype=tf.float32)
    filter_v = tf.constant(filter_v, dtype=tf.float32)
    diff_h = tf.nn.conv2d(img, filter_h, strides=[1, 1, 1, 1], padding='SAME')
    diff_v = tf.nn.conv2d(img, filter_v, strides=[1, 1, 1, 1], padding='SAME')
    diff = tf.divide(tf.abs(diff_h) + tf.abs(diff_v), tf.constant(8, dtype=tf.float32))
    # diff = tf.divide(tf.abs(diff_h) + tf.abs(diff_v), tf.constant(8 * 255, dtype=tf.float32))
    diff = tf.reshape(diff, shape)
    return diff


def resnet_caffe_preprocessing_rbg(feature, label):
    a = np.array([123.68, 116.779, 103.939])
    a = a[[0, 2, 1]]
    imagenet_mean = tf.constant(a, dtype=tf.float32)
    feature = tf.subtract(feature, imagenet_mean)
    return feature, label


def resnet_caffe_preprocessing(feature, label):
    a = np.array([123.68, 116.779, 103.939])
    if data_advance == 'color_sw_RBG':
        a = a[[0, 2, 1]]
    imagenet_mean = tf.constant(a, dtype=tf.float32)
    feature = tf.subtract(feature, imagenet_mean)
    return feature, label


def resnet_caffe_preprocessing_reverse(feature, label):
    a = np.array([123.68, 116.779, 103.939])
    if data_advance == 'color_sw_RBG':
        a = a[[0, 2, 1]]
    imagenet_mean = a
    feature = feature + imagenet_mean
    return feature, label


def resnet_tf_preprocessing(feature, label):
    feature = tf.divide(feature, tf.constant(127.5))
    feature = tf.subtract(feature, tf.constant(1.0))
    return feature, label


def resnet_0_1_preprocessing(feature, label):
    feature = tf.divide(feature, tf.constant(255, dtype=tf.float32))
    return feature, label


def tmp_combine_6ch(feature, label, config):
    diff = color_diff_121(feature)
    diff = tf.slice(diff, [1, 1, 0], [config['crop_h'], config['crop_w'], config['channel']])
    feature = tf.slice(feature, [1, 1, 0], [config['crop_h'], config['crop_w'], config['channel']])
    # feature, _ = resnet_0_1_preprocessing(feature, label)
    feature = tf.concat([feature, diff], 2)
    return feature, label


# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     # Disable first GPU
#     tf.config.set_visible_devices(physical_devices[1:], 'GPU')
#     logical_devices = tf.config.list_logical_devices('GPU')
#     # Logical device was not created for first GPU
#     assert len(logical_devices) == len(physical_devices) - 1
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass


train_config = {'crop_h': train_crop_h,
                'crop_w': train_crop_w,
                'resize_short_edge_max': train_resize_short_edge_max,
                'resize_short_edge_min': train_resize_short_edge_min,
                'channel': channel
                }
val_config = {'crop_h': val_crop_h,
              'crop_w': val_crop_w,
              'resize_short_edge_max': val_resize_short_edge_max,
              'resize_short_edge_min': val_resize_short_edge_min,
              'channel': channel
              }

tf.random.set_seed(seed=random_seed)
train_files_list = tf.data.Dataset.list_files(str(Path(train_dir).parent.joinpath('train.tfrecord*')))
val_files_list = tf.data.Dataset.list_files(str(Path(val_dir).parent.joinpath('val.tfrecord*')))
# val_files_list = tf.data.Dataset.list_files(
#     str(Path(val_dir).parent.parent.joinpath('color_sw_RBG').joinpath('val.tfrecord*')))
# for f in train_files_list.take(5):
#     print(f.numpy())
train_dataset = tf.data.TFRecordDataset(train_files_list)
val_dataset = tf.data.TFRecordDataset(val_files_list)
# count = 0
# for serialized_example in val_dataset.take(-1):
#     # example = tf.train.Example()
#     # example.ParseFromString(serialized_example.numpy())
#     # x_1 = np.array(example.features.feature['feature'].float_list.value)
#     # y_1 = np.array(example.features.feature['label'].int64_list.value)
#     print(count)
#     count = count + 1

# train_dataset.apply(tf.data.experimental.assert_cardinality(train_cardinality))
# val_dataset.apply(tf.data.experimental.assert_cardinality(val_cardinality))
train_dataset = train_dataset.map(example_parse_decode)
val_dataset = val_dataset.map(example_parse_decode)
train_dataset = train_dataset.map(lambda img, label: random_crop(img, label, train_config))
val_dataset = val_dataset.map(lambda img, label: random_crop(img, label, val_config))
train_dataset = train_dataset.map(resnet_caffe_preprocessing)
val_dataset = val_dataset.map(resnet_caffe_preprocessing)

# train_dataset = train_dataset.map(lambda img, label: random_plus2_crop(img, label, train_config))
# val_dataset = val_dataset.map(lambda img, label: random_plus2_crop(img, label, val_config))
# train_dataset = train_dataset.map(lambda img, label: tmp_combine_6ch(img, label, train_config))
# val_dataset = val_dataset.map(lambda img, label: tmp_combine_6ch(img, label, val_config))

# train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
# val_dataset = val_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(train_batch_size)
val_dataset = val_dataset.batch(val_batch_size)
train_dataset = train_dataset.repeat(10)
val_dataset = val_dataset.repeat(10)

base_model = load_model(str(Path(ckpt_dir).parent.parent.joinpath('none/random_crop')) + '/lr1e-1-ckpt-epoch0278_loss-0.0835_accuracy-0.9747_val_loss-3.1642_val_accuracy-0.7815')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True, rankdir="TB", expand_nested=False, dpi=96)
if not Path.is_dir(Path(ckpt_dir + 'data')):
    Path.mkdir(Path(ckpt_dir + 'data'))
count = 0
feature_list = []
label_list = []
for element in val_dataset.as_numpy_iterator():
    predict = model.predict(element[0])
    label = element[1].argmax(1)
    reversed_element = resnet_caffe_preprocessing_reverse(element[0], element[1])
    for i in range(len(label)):
        if not Path.is_dir(Path(ckpt_dir + 'data/{:0>4d}'.format(label[i]))):
            Path.mkdir(Path(ckpt_dir + 'data/{:0>4d}'.format(label[i])))
        feature_list.append(predict[i])
        label_list.append(label[i])
        img = np.array(reversed_element[0][i], dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(ckpt_dir + 'data/{:0>4d}/{:0>10d}.jpg'.format(label[i], count), img)
        count += 1
    _ = gc.collect()
    print(count)
features = np.zeros((len(feature_list),) + feature_list[0].shape)
for i in range(len(feature_list)):
    features[i, ...] = feature_list[i]
labels = np.zeros((len(label_list),) + label_list[0].shape)
for i in range(len(label_list)):
    labels[i, ...] = label_list[i]
np.save(ckpt_dir + 'data/features', features)
np.save(ckpt_dir + 'data/labels', labels)
