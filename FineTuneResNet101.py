import warnings

warnings.filterwarnings('ignore')

import numpy as np
import os
import tensorflow as tf
import cv2
import random
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Import data
# change the dataset here###
dataset = 'AWA2'
# datatype: img, tfrecord
datatype = 'img'
# data data_advance: color_diff_121, none
data_advance = 'none'
# preprocess
preprocess = 'caffe'
# crop
crop_type = 'random'
##############################

batch_size = 128
train_dir = '/media/uscc/HDD2/Dataset/{}/{}/{}/train/'.format(dataset, datatype, data_advance)
val_dir = '/media/uscc/HDD2/Dataset/{}/{}/{}/val/'.format(dataset, datatype, data_advance)
IMG_SHAPE = 224


if dataset == 'SUN':
    class_attr_shape = (102,)
    class_attr_dim = 102
    class_num = 717
    seen_class_num = 645
    unseen_class_num = 72
elif dataset == 'CUB':
    class_attr_shape = (312,)
    class_attr_dim = 312
    class_num = 200
    seen_class_num = 150
    unseen_class_num = 50
elif dataset == 'AWA2':
    class_attr_shape = (85,)
    class_attr_dim = 85
    class_num = 50
    seen_class_num = 40
    unseen_class_num = 10
    file_type = '.jpg'
    train_cardinality = 58176
    val_cardinality = 15872
elif dataset == 'plant':
    class_attr_shape = (46,)
    class_attr_dim = 46
    class_num = 38
    seen_class_num = 25
    unseen_class_num = 13
elif dataset == 'imagenet':
    class_num = 1000
    seen_class_num = 1000
    file_type = '.JPEG'
    train_cardinality = 1281167
    val_cardinality = 50000


# 考慮 shuffle every epoch
def img_generator(target_directory, color_mode, shuffle=False):
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
    print("Found {} images belonging to {} classes.".format(file_num, class_count))
    if shuffle:
        new_instance_list = []
        while len(instance_list):
            rand_num = random.randint(0, len(instance_list) - 1)
            new_instance_list.append(instance_list[rand_num])
            del instance_list[rand_num]
        instance_list = new_instance_list
    while True:
        for instance in instance_list:
            img = cv2.imread(instance['path'])

            if color_mode == "RGB":
                img = img[..., ::-1]
            img = np.array(img, dtype=np.float32)
            label = np.zeros(seen_class_num, dtype=np.float32)
            label[instance['label']] = 1
            yield img, label


# 考慮加入 last batch 捨去功能, rotate
def crop_generator(target_directory, batch_size=1, crop_type=None, crop_w=256, crop_h=256, resize_short_edge_max=256,
                   resize_short_edge_min=256, horizontal_flip=False, color_mode="BGR", shuffle=True, seed=0):
    img_gen = img_generator(target_directory, shuffle=shuffle, color_mode=color_mode)
    random.seed(seed)
    while True:
        batch_feature = np.empty([0, crop_h, crop_w, 3], dtype=np.float32)
        batch_label = np.empty([0, seen_class_num], dtype=np.float32)
        for i in range(batch_size):
            img, label = next(img_gen)
            if horizontal_flip:
                if random.randint(0, 1):
                    img = img[:, ::-1, :]
            height, width, _ = img.shape
            if height < width:
                new_height = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_width = round(width*new_height/height)
            else:
                new_width = random.randint(resize_short_edge_min, resize_short_edge_max)
                new_height = round(height * new_width / width)
            img = cv2.resize(img, (new_width, new_height))
            if crop_type == None:
                crop = cv2.resize(img, (crop_w, crop_h))
            elif crop_type == "random":   # 先不處理 crop 過大的情況
                y0 = random.randint(0, new_height - crop_h)
                x0 = random.randint(0, new_width - crop_w)
                y1 = y0 + crop_h
                x1 = x0 + crop_w
                crop = img[y0:y1, x0:x1, :]
            else:
                crop = cv2.resize(img, (crop_w, crop_h))    # center crop 尚未完成
            if preprocess == "caffe":
                if color_mode == 'BGR':
                    crop = crop - np.array([[[103.939, 116.779, 123.68]]], dtype=np.float32)
                elif color_mode == 'RGB':
                    crop = crop - np.array([123.68, 116.779, 103.939], dtype=np.float32)
            crop = crop[np.newaxis, ...]
            label = label[np.newaxis, ...]
            batch_feature = np.concatenate((batch_feature, crop), 0)
            batch_label = np.concatenate((batch_label, label), 0)

        # x = batch_x.shape[1] // 2
        # y = batch_x.shape[2] // 2
        # size = new_size // 2
        # yield batch_x[:, x - size:x + size, y - size:y + size], batch_y
        yield batch_feature, batch_label


train_data_gen = crop_generator(
    train_dir,
    batch_size=batch_size,
    crop_type=crop_type,
    crop_w=IMG_SHAPE,
    crop_h=IMG_SHAPE,
    resize_short_edge_max=480,
    resize_short_edge_min=256,
    shuffle=True,
    color_mode="BGR",
    seed=486
)

val_data_gen = crop_generator(
    val_dir,
    batch_size=batch_size,
    crop_type=crop_type,
    crop_w=IMG_SHAPE,
    crop_h=IMG_SHAPE,
    resize_short_edge_max=480,
    resize_short_edge_min=256,
    shuffle=False,
    color_mode="BGR",
    seed=486
)

# a = next(train_data_gen)
# image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
# image_gen = ImageDataGenerator()
# image_gen = ImageDataGenerator()
# train_data_gen = image_gen.flow_from_directory(
#     batch_size=batch_size,
#     directory=train_dir,
#     shuffle=True,
#     color_mode="rgb",
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     seed=42
# )

# image_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
# image_gen_val = ImageDataGenerator()
# val_data_gen = image_gen.flow_from_directory(
#     batch_size=batch_size,
#     directory=val_dir,
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     color_mode="rgb",
#     seed=42
# )

# class_weights = class_weight.compute_class_weight(
#            'balanced',
#             np.unique(train_data_gen.classes),
#             train_data_gen.classes)


## Fine tune or Retrain ResNet101
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, 3))

# # lock the model
# for layer in base_model.layers:
#     layer.trainable = False

# add a global averge pollinf layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

# add a dense
# x = Dense(1024, activation='relu')(x)

# add a classifier
    predictions = Dense(seen_class_num, activation='softmax')(x)

# Constructure
    model = Model(inputs=base_model.input, outputs=predictions)

# compile
# model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#               , loss='categorical_crossentropy',metrics=['accuracy'])
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                            rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖


early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,
                               verbose=1)
model_checkpoint = ModelCheckpoint('/media/uscc/SSD/NE6091069/p_learning/model/{}/{}/{}/{}_crop/'.format(dataset, datatype, data_advance, crop_type),
                                   save_weights_only=True,
                                   save_freq='epoch',
                                   verbose=1)
reduce_LR_on_plateau = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=5,
                                         verbose=1,
                                         min_delta=1,
                                         min_lr=0.00001)

STEP_SIZE_TRAIN = train_cardinality // batch_size
STEP_SIZE_VALID = val_cardinality // batch_size

epochs = 100
model.compile(optimizer=SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    # class_weight=class_weights,
                    callbacks=[model_checkpoint, reduce_LR_on_plateau]
                    )
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
