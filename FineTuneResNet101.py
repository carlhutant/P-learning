import warnings
warnings.filterwarnings('ignore')

import configure
import numpy as np
import os
import tensorflow as tf
import cv2
import random
import math
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

# datatype: img, tfrecord
# data_advance: color_diff_121, none
# preprocess: caffe
# color_mode: BGR, RGB, none
dataset = 'AWA2'
datatype = 'img'
data_advance = 'none'
preprocess = 'caffe'
color_mode = "BGR"

train_shuffle = True
train_shuffle_every_epoch = True
val_shuffle = False
val_shuffle_every_epoch = False
random_seed = 486

batch_size = 16
train_final_batch_opt = 'complete'
val_final_batch_opt = 'complete'
train_resize_short_edge_max = 480
train_resize_short_edge_min = 256
train_IMG_SHAPE = 224

train_horizontal_flip = True
train_crop_type = 'random'
train_crop_w = train_IMG_SHAPE
train_crop_h = train_IMG_SHAPE

multi_GPU = False
GPU_memory_growth = False


dataset_dir = configure.dataset_dir
model_dir = configure.model_dir
train_dir = '{}/{}/{}/{}/train/'.format(dataset_dir, dataset, datatype, data_advance)
val_dir = '{}/{}/{}/{}/val/'.format(dataset_dir, dataset, datatype, data_advance)
ckpt_dir = '{}/{}/{}/{}/{}_crop/'.format(model_dir, dataset, datatype, data_advance, train_crop_type)
model_save_path = '{}/{}/{}/{}/{}_crop/final/'.format(model_dir, dataset, datatype, data_advance, train_crop_type)


if GPU_memory_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if dataset == 'AWA2':
    class_num = 40
    file_type = '.jpg'
    train_cardinality = 24264
    val_cardinality = 6070
elif dataset == 'imagenet':
    class_num = 1000
    file_type = '.JPEG'
    train_cardinality = 1281167
    val_cardinality = 50000
else:
    raise RuntimeError


# 考慮 shuffle every epoch
def img_generator(target_directory, shuffle, shuffle_every_epoch):
    walk_generator = os.walk(target_directory)
    root, directory, _ = next(walk_generator)
    instance_list = []
    class_count = 0
    for d in directory:
        walk_generator2 = os.walk(root + d)
        flies_root, _, files = next(walk_generator2)
        for file in files:
            if file.endswith(file_type):
                instance_list.append({'path': os.path.join(flies_root, file), 'label': class_count})
        class_count = class_count + 1
    file_num = len(instance_list)
    if shuffle or shuffle_every_epoch:
        random.shuffle(instance_list)
    yield file_num, class_count
    while True:
        for i in range(file_num):
            instance = instance_list[i]
            img = cv2.imread(instance['path'])
            if color_mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
            label = np.zeros(class_num, dtype=np.float32)
            label[instance['label']] = 1
            yield img, label
        if shuffle_every_epoch:
            random.shuffle(instance_list)


###############################################
# target_directory: 目錄
# batch_size
# crop_type: random, center, none
# crop_h: height
# crop_w: width
# resize_short_edge_max
# resize_short_edge_min
# horizontal_flip: True, False
# color_mode='BGR', 'RGB'
# shuffle: True, False
# seed: random seed
###############################################
# 考慮加入 last batch 捨去功能, rotate
def crop_generator(target_directory, final_batch_opt, crop_type, crop_h, crop_w, resize_short_edge_max,
                   resize_short_edge_min, horizontal_flip, shuffle, shuffle_every_epoch):
    img_gen = img_generator(target_directory, shuffle, shuffle_every_epoch)
    file_num, dir_num = next(img_gen)
    print("Found {} images belonging to {} classes.".format(file_num, dir_num))
    if file_num != train_cardinality and file_num != val_cardinality:
        raise RuntimeError
    if dir_num != class_num:
        raise RuntimeError
    yield
    random.seed(random_seed)
    file_remain_num = file_num
    while True:
        batch_feature = np.empty([0, crop_h, crop_w, 3], dtype=np.float32)
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
            if horizontal_flip:
                if random.randint(0, 1):
                    img = img[:, ::-1, :]
            height, width, _ = img.shape
            if height < width:
                new_height = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_width = round(width * new_height / height)
            else:
                new_width = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_height = round(height * new_width / width)
            img = cv2.resize(img, (new_width, new_height))
            if crop_type == 'center':  # center crop 尚未完成
                crop = cv2.resize(img, (crop_w, crop_h))
            elif crop_type == 'random':  # 先不處理 crop 大於原圖的情況
                y0 = random.randint(0, new_height - crop_h)
                x0 = random.randint(0, new_width - crop_w)
                y1 = y0 + crop_h
                x1 = x0 + crop_w
                crop = img[y0:y1, x0:x1, :]
            else:
                crop = cv2.resize(img, (crop_w, crop_h))
            if preprocess == 'caffe':
                if color_mode == 'BGR':
                    crop = crop - np.array([[[103.939, 116.779, 123.68]]], dtype=np.float32)
                elif color_mode == 'RGB':
                    crop = crop - np.array([123.68, 116.779, 103.939], dtype=np.float32)
            crop = crop[np.newaxis, ...]
            label = label[np.newaxis, ...]
            batch_feature = np.concatenate((batch_feature, crop), 0)
            batch_label = np.concatenate((batch_label, label), 0)
        yield batch_feature, batch_label
        file_remain_num = file_remain_num - batch_size
        if file_remain_num<1:
            file_remain_num = file_num


train_data_gen = crop_generator(
    target_directory=train_dir,
    final_batch_opt=train_final_batch_opt,
    crop_type=train_crop_type,
    crop_h=train_crop_h,
    crop_w=train_crop_w,
    resize_short_edge_max=train_resize_short_edge_max,
    resize_short_edge_min=train_resize_short_edge_min,
    horizontal_flip=train_horizontal_flip,
    shuffle=train_shuffle,
    shuffle_every_epoch=train_shuffle_every_epoch
)
next(train_data_gen)
val_data_gen = crop_generator(
    target_directory=val_dir,
    final_batch_opt=val_final_batch_opt,
    crop_type=train_crop_type,
    crop_w=train_crop_h,
    crop_h=train_crop_w,
    resize_short_edge_max=train_resize_short_edge_max,
    resize_short_edge_min=train_resize_short_edge_min,
    horizontal_flip=train_horizontal_flip,
    shuffle=val_shuffle,
    shuffle_every_epoch=val_shuffle_every_epoch
)
next(val_data_gen)

# # test final_batch_opt
# a = next(train_data_gen)
# file_remain_num = train_cardinality-batch_size
# batch_data_num = min(batch_size, file_remain_num)
# count = 0
# while file_remain_num > 0:
#     count = count + 1
#     print(count)
#     b = next(train_data_gen)
#     file_remain_num = file_remain_num - batch_size
# c = next(train_data_gen)

# Fine tune or Retrain ResNet101
if multi_GPU:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
        # add a global average pooling layer
        x = GlobalAveragePooling2D()(base_model.output)
        # # add a dense
        # x = Dense(1024, activation='relu')(x)
        # add a classifier
        predictions = Dense(class_num, activation='softmax')(x)
        # Construction
        model = Model(inputs=base_model.input, outputs=predictions)
else:
    base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))
    # add a global average pooling layer
    x = GlobalAveragePooling2D()(base_model.output)
    # # add a dense
    # x = Dense(1024, activation='relu')(x)
    # add a classifier
    predictions = Dense(class_num, activation='softmax')(x)
    # Construction
    model = Model(inputs=base_model.input, outputs=predictions)

# keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                            rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖


early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(ckpt_dir + 'ckpt-epoch{epoch:04d}'
                                            + '_loss-{loss:.4f}'
                                            + '_accuracy-{accuracy:.4f}'
                                            + '_val_loss-{val_loss:.4f}'
                                            + '_val_accuracy-{val_accuracy:.4f}',
                                   save_weights_only=False,
                                   save_freq='epoch',
                                   verbose=0)
reduce_LR_on_plateau = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=10,
                                         verbose=1,
                                         min_delta=1,
                                         min_lr=0.00001)

STEP_SIZE_TRAIN = math.ceil(train_cardinality / batch_size)
STEP_SIZE_VALID = math.ceil(val_cardinality / batch_size)

epochs = 2000
# try:
#     model = tf.keras.models.load_model(ckpt_dir)
#     # model.load_weights(ckp_path)
#     print('check point found.')
# except:
#     print('no check point found.')

model.compile(optimizer=SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=False)
              , loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[model_checkpoint]
                    )
# model.save(model_save_path)
# epochs = 10
#
# for layer in model.layers[:335]:
#     layer.trainable = False
# for layer in model.layers[335:]:
#     layer.trainable = True
#
#
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#
# STEP_SIZE_TRAIN = train_data_gen.n // train_data_gen.batch_size
# STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size
#
# model.fit_generator(train_data_gen,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     epochs=epochs,
#                     validation_data=val_data_gen,
#                     validation_steps=STEP_SIZE_VALID,
#                     #                     class_weight=class_weights,
#                     callbacks=[early_stopping]
#                     )
#
# model.save('./model/{}/{}_{}/ResNet101_step2.h5'.format(dataset, data_advance, datatype))

# # Evaluate
# model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#               , loss='categorical_crossentropy', metrics=['accuracy'])
# # model.load_weights("./model/AWA2/FineTuneResNet101_edge_with_head.h5")
# # STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size
# score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
# print(score)
