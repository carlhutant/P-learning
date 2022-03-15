import warnings
import os
import tensorflow as tf
import numpy as np
import cv2
import random
import math

import tensorflow.keras.models

import ResnetDIY

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from configure import *
from pathlib import Path

if GPU_memory_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


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


def color_diff_121_abs_3ch(img):
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


def diy_normalizing(feature, label):
    if dataset == 'imagenet':
        if data_advance == 'none':
            mean = np.array([122.6774804, 116.65180841, 103.99943435])
            std = np.array([59.53750016, 58.50399986, 58.71025873])
        else:
            raise RuntimeError
    elif dataset == 'AWA2':
        if data_advance == 'none':
            mean = np.array([119.09687091, 119.04670833, 100.78114277])
            std = np.array([55.85246336, 54.56388751, 54.51621827])
        elif data_advance == 'color_sw_RBG':
            mean = np.array([119.09687091, 100.78114277, 119.04670833])
            std = np.array([55.85246336, 54.51621827, 54.56388751])
        elif data_advance == 'color_diff_121_abs_3ch':
            mean = np.array([11.03982951, 10.86760973, 10.90499236])
            std = np.array([11.97321782, 11.92262023, 11.88327279])
        else:
            raise RuntimeError
    else:
        raise RuntimeError
    awa2_mean = tf.constant(mean, dtype=tf.float32)
    awa2_std = tf.constant(std, dtype=tf.float32)
    feature = tf.subtract(feature, awa2_mean)
    feature = tf.divide(feature, awa2_std)
    return feature, label


def diy_normalizing_multi_domain(feature):
    if data_advance == 'none':
        mean1 = np.array([119.09687091, 119.04670833, 100.78114277])
        std1 = np.array([55.85246336, 54.56388751, 54.51621827])
    elif data_advance == 'color_diff_121_abs_3ch':
        mean1 = np.array([11.03982951, 10.86760973, 10.90499236])
        std1 = np.array([11.97321782, 11.92262023, 11.88327279])
    elif data_advance == 'color_sw_RBG':
        mean1 = np.array([119.09687091, 119.04670833, 100.78114277])
        std1 = np.array([55.85246336, 54.56388751, 54.51621827])
    else:
        raise RuntimeError
    if data_advance2 == 'none':
        mean2 = np.array([119.09687091, 119.04670833, 100.78114277])
        std2 = np.array([55.85246336, 54.56388751, 54.51621827])
    elif data_advance2 == 'color_diff_121_abs_3ch':
        mean2 = np.array([11.03982951, 10.86760973, 10.90499236])
        std2 = np.array([11.97321782, 11.92262023, 11.88327279])
    else:
        raise RuntimeError
    mean = np.concatenate((mean1, mean2), axis=0)
    std = np.concatenate((std1, std2), axis=0)
    awa2_mean = tf.constant(mean, dtype=tf.float32)
    awa2_std = tf.constant(std, dtype=tf.float32)
    feature = tf.subtract(feature, awa2_mean)
    feature = tf.divide(feature, awa2_std)
    return feature


def resnet_caffe_preprocessing(feature, label):
    a = np.array([123.68, 116.779, 103.939])
    if data_advance == 'color_sw_RBG':
        a = a[[0, 2, 1]]
    imagenet_mean = tf.constant(a, dtype=tf.float32)
    feature = tf.subtract(feature, imagenet_mean)
    return feature, label


def resnet_tf_preprocessing(feature, label):
    feature = tf.divide(feature, tf.constant(127.5))
    feature = tf.subtract(feature, tf.constant(1.0))
    return feature, label


def resnet_0_1_preprocessing(feature, label):
    feature = tf.divide(feature, tf.constant(255, dtype=tf.float32))
    return feature, label


def tmp_combine_6ch(feature, label, config):
    # if data_advance != 'none' or data_advance2 != 'color_diff_121_abs_3ch':
    #     raise RuntimeError
    diff = color_diff_121_abs_3ch(feature)
    diff = tf.slice(diff, [1, 1, 0], [config['crop_h'], config['crop_w'], config['channel']])
    feature = tf.slice(feature, [1, 1, 0], [config['crop_h'], config['crop_w'], config['channel']])
    # feature = np.zeros((config['crop_h'], config['crop_w'], config['channel']), dtype=np.double)
    # feature = tf.constant(feature, tf.float32)
    feature = tf.concat([feature, diff], 2)
    feature = diy_normalizing_multi_domain(feature)
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
mode = "train"

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

train_dataset.apply(tf.data.experimental.assert_cardinality(train_cardinality))
val_dataset.apply(tf.data.experimental.assert_cardinality(val_cardinality))
train_dataset = train_dataset.map(example_parse_decode)
val_dataset = val_dataset.map(example_parse_decode)
train_dataset = train_dataset.map(lambda img, label: random_crop(img, label, train_config))
val_dataset = val_dataset.map(lambda img, label: random_crop(img, label, val_config))
train_dataset = train_dataset.map(diy_normalizing)
val_dataset = val_dataset.map(diy_normalizing)

# train_dataset = train_dataset.map(lambda img, label: random_plus2_crop(img, label, train_config))
# val_dataset = val_dataset.map(lambda img, label: random_plus2_crop(img, label, val_config))
# train_dataset = train_dataset.map(lambda img, label: tmp_combine_6ch(img, label, train_config))
# val_dataset = val_dataset.map(lambda img, label: tmp_combine_6ch(img, label, val_config))

train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
val_dataset = val_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(train_batch_size)
val_dataset = val_dataset.batch(val_batch_size)
train_dataset = train_dataset.repeat()
val_dataset = val_dataset.repeat()

# take = train_dataset.take(10)
# a = take.as_numpy_iterator()
#
# for _ in range(10):
#     b = next(a)
#     for i in range(train_batch_size):
#         c = b[0][i, ..., :3]
#         c = c[..., [2, 1, 0]]
#         sh = np.array(c, dtype=np.uint8)
#         cv2.imshow('123', sh)
#         c2 = b[0][i, ..., 3:]
#         c2 = c2[..., [2, 1, 0]]
#         sh = np.array(c2, dtype=np.uint8)
#         cv2.imshow('321', sh)
#         cv2.waitKey()

# # Fine tune or Retrain ResNet101
# import resnet
# base_model = ResNet101(weights='imagenet', include_top=True)
# model = load_model("/media/uscc/HDD1/carl/Model/AWA2/tfrecord/none_color_diff_121_abs_3ch/random_crop/no_dropout/lr1e-1-ckpt-epoch0050_loss-15.3944_accuracy-0.9452_val_loss-19.5259_val_accuracy-0.7952")
model = load_model(ckpt_dir + "lr1e-2/lr1e-2-ckpt-epoch0016_loss-0.7403_accuracy-0.8164_val_loss-1.9177_val_accuracy-0.6218")
# model = load_model("/media/uscc/HDD1/carl/Model/AWA2/tfrecord/none/random_crop/lr1e-1-ckpt-epoch0015_loss-1.6007_accuracy-0.6290_val_loss-2.7365_val_accuracy-0.5600")
# model1 = load_model("/media/uscc/HDD1/carl/Model/AWA2/tfrecord/none/random_crop/lr1e-1-ckpt-epoch0200_loss-0.1270_accuracy-0.9619_val_loss-5.9098_val_accuracy-0.7719")
# model2 = load_model("/media/uscc/HDD1/carl/Model/AWA2/tfrecord/color_diff_121_abs_3ch/random_crop/lr1e-1-ckpt-epoch0200_loss-0.1668_accuracy-0.9496_val_loss-2.7433_val_accuracy-0.7142")
# model1 = Model(inputs=model1.input, outputs=model1.layers[-2].output)
# model2 = Model(inputs=model2.input, outputs=model2.layers[-2].output)
# inputs = Input(shape=(224, 224, 6))
# gx1 = model1(inputs[..., :3])
# gx2 = model2(inputs[..., 3:])
# x = Concatenate()([gx1, gx2])
# x = Dropout(rate=0.5)(x)
# outputs = Dense(units=class_num, activation='softmax')(x)
# model = Model(inputs=inputs, outputs=outputs)
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                           rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖

# model = ResnetDIY.resnet101(class_num=class_num, channel=channel)
# model = ResnetDIY.resnet101(class_num=class_num, channel=6)
# base_model = tensorflow.keras.models.load_model(ckpt_dir + 'lr1e-4/lr1e-4-ckpt-epoch0059_loss-0.0603_accuracy-0.9831_val_loss-4.1679_val_accuracy-0.7912')
# model = Model(inputs=model.input, outputs=model.layers[-3].output)
# add a global average pooling layer
# x = base_model.layers[-3].output
# x = GlobalAveragePooling2D()(x)
# add a classifier
# predictions = Dense(class_num, activation='softmax')(x)
# Construct
# model = Model(inputs=base_model.input, outputs=predictions)
# base_model.save('E:/Model/AWA2/tfrecord/none/imagenet')
# model = load_model('E:/Model/AWA2/tfrecord/none/imagenet')
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                           rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖

# for layer in model.layers[:-1]:
#     layer.trainable = False
# for layer in model.layers[-1:]:
#     layer.trainable = True

model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.5, nesterov=False),
              loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = math.ceil(train_cardinality // train_batch_size)
STEP_SIZE_VALID = math.ceil(val_cardinality // val_batch_size)

# STEP_SIZE_TRAIN = 20
# STEP_SIZE_VALID = 20

model_checkpoint = ModelCheckpoint(ckpt_dir + 'lr1e-3-ckpt-epoch{epoch:04d}'
                                   + '_loss-{loss:.4f}'
                                   + '_accuracy-{accuracy:.4f}'
                                   + '_val_loss-{val_loss:.4f}'
                                   + '_val_accuracy-{val_accuracy:.4f}',
                                   save_weights_only=False,
                                   save_freq='epoch',
                                   verbose=0)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

epochs = 300
# print(model.layers[-1].name)
# print(model.layers[-6].name)
# a = model.layers[-1].weights[0][0]
# a2 = model.layers[-6].weights[0][0]
model.fit(train_dataset,
          epochs=epochs,
          steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=val_dataset,
          validation_steps=STEP_SIZE_VALID,
          callbacks=[model_checkpoint]
          )
# model.evaluate(val_dataset, batch_size=val_batch_size)
# b = model.layers[-1].weights[0][0]
# b2 = model.layers[-6].weights[0][0]
# print(a == b)
# print(a2 == b2)
# c = 0
# model.compile(optimizer=SGD(learning_rate=0, momentum=0.5, nesterov=False)
#               , loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(train_dataset,
#           epochs=epochs,
#           steps_per_epoch=STEP_SIZE_TRAIN,
#           validation_data=val_dataset,
#           validation_steps=STEP_SIZE_VALID,
#           # callbacks=[model_checkpoint]
#           )
# c = model.layers[-1].weights[0][0]
# c = 0
# model.save('./model/{}/none_finetune_tfrecord/ResNet101_none_step1_epoch{}.h5'.format(dataset, i))

# for layer in model.layers[:335]:
#     layer.trainable = False
# for layer in model.layers[335:]:
#     layer.trainable = True
#
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#
# STEP_SIZE_TRAIN = train_cardinality // train_batch_size
# STEP_SIZE_VALID = val_cardinality // val_batch_size
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# epochs = 10
# print("step 2:")
# for i in range(epochs):
#     print("step 2 epoch {}:".format(i + 1))
#     model.fit(train_dataset, epochs=1, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=val_dataset,
#               validation_steps=STEP_SIZE_VALID, callbacks=[early_stopping])
#     # model.save('./model/{}/none_finetune_tfrecord/ResNet101_none_step2_epoch{}.h5'.format(dataset, i))

# ## Evaluate
# model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#               , loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights("./model/AWA2/FineTuneResNet101_edge_with_head.h5")
# STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size
# score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
# print(score)
