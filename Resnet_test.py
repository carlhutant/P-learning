import warnings
import numpy as np
import os
import gc
import tensorflow as tf
import cv2
import random
import math
from pathlib import Path
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from data_generator import *

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # val_data_gen = crop_generator(
    #     target_directory=val_dir,
    #     batch_size=val_batch_size,
    #     final_batch_opt=val_final_batch_opt,
    #     crop_type=val_crop_type,
    #     crop_w=val_crop_h,
    #     crop_h=val_crop_w,
    #     resize_short_edge_max=val_resize_short_edge_max,
    #     resize_short_edge_min=val_resize_short_edge_min,
    #     horizontal_flip=val_horizontal_flip,
    #     shuffle=val_shuffle,
    #     shuffle_every_epoch=val_shuffle_every_epoch
    # )
    # next(val_data_gen)

    val_data_gen = parallel_batch_generator(
        target_directory=val_dir,
        batch_size=val_batch_size,
        final_batch_opt=val_final_batch_opt,
        crop_type=val_crop_type,
        crop_w=val_crop_h,
        crop_h=val_crop_w,
        resize_short_edge_max=val_resize_short_edge_max,
        resize_short_edge_min=val_resize_short_edge_min,
        horizontal_flip=val_horizontal_flip,
        shuffle=val_shuffle,
        shuffle_every_epoch=val_shuffle_every_epoch
    )
    next(val_data_gen)

    # test generator speed
    count = 0
    print('Start testing generator speed')
    while True:
        a = next(val_data_gen)
    #     # img = np.array(a[0][0, ...], dtype=np.uint8)
    #     # cv2.imshow('123', img)
    #     # cv2.waitKey()
        count = count + 1
        # print(count)

    if GPU_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if multi_GPU:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, channel))
            # add a global average pooling layer
            x = GlobalAveragePooling2D()(base_model.output)
            # # add a dense
            # x = Dense(1024, activation='relu')(x)
            # add a classifier
            predictions = Dense(class_num, activation='softmax')(x)
            # Construction
            model = Model(inputs=base_model.input, outputs=predictions)
    else:
        base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, channel))
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

    STEP_SIZE_VALID = math.ceil(val_cardinality / val_batch_size)

    print('resize_short_edge={}~{}'.format(val_resize_short_edge_min, val_resize_short_edge_max))
    # target_dir = Path(model_dir).joinpath('AWA2').joinpath('img').joinpath('none').joinpath('random_crop').joinpath('test')
    # target_dir = ckpt_dir
    # walk_generator = os.walk(target_dir)
    # root, directories, files = next(walk_generator)
    # files.sort()
    # for f in files:
    #     print(f)
    # for f in files:
    #     if f.startswith('ckpt-epoch') and f.endswith('index'):
    #         try:
    #             # model = tf.keras.models.load_model(ckp_path)
    #             # model = tf.keras.models.load_model('D:\\Download\\P_learning\\model\\AWA2\img\\none\\random_crop\\ckpt-epoch0001_loss-1.6212_accuracy-0.5446_val_loss-3.0151_val_accuracy-0.5061')
    #             # model.load_weights(ckp_path)
    #             # model.load_weights('D:\\Download\\P_learning\\model\\AWA2\img\\none\\random_crop\\ckpt-epoch0031_loss-1.6706_accuracy-0.5280_val_loss-1.9804_val_accuracy-0.5219')
    #             model.load_weights(Path(root).joinpath(Path(f).stem))
    #             print(str(Path(f).stem))
    #             # print('check point found.')
    #         except:
    #             # print('no check point found.')
    #             raise RuntimeError
    #         # Evaluate
    #         # score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
    #         # print(score)
    #         total_count = 0
    #         positive_count = 0
    #         # step_count = 0
    #         for step in range(STEP_SIZE_VALID):
    #             batch_instance = next(val_data_gen)
    #             batch_predict = model.predict(batch_instance[0])
    #             if batch_instance[0].shape[0] % 10 == 0:
    #                 batch_instance_num = int(batch_instance[0].shape[0]/10)
    #             else:
    #                 raise RuntimeError
    #             for instance_No in range(batch_instance_num):
    #                 instance_predict = batch_predict[instance_No*10:instance_No*10+10:, ...]
    #                 instance_predict = instance_predict.sum(axis=0)
    #                 maxarg = instance_predict.argmax(axis=0)
    #                 if batch_instance[1][instance_No*10][maxarg] == 1:
    #                     positive_count = positive_count + 1
    #                 total_count = total_count + 1
    #             # print('{}/{}-{}/{}'.format(step_count, STEP_SIZE_VALID, positive_count, total_count))
    #             # step_count += 1
    #         print(positive_count/total_count)
    # a = 0

    try:
        # model = tf.keras.models.load_model(ckp_path)
        # model = tf.keras.models.load_model('D:\\Download\\P_learning\\model\\AWA2\img\\none\\random_crop\\ckpt-epoch0001_loss-1.6212_accuracy-0.5446_val_loss-3.0151_val_accuracy-0.5061')
        # model.load_weights(ckp_path)
        # model.load_weights('D:\\Download\\P_learning\\model\\AWA2\img\\none\\random_crop\\ckpt-epoch0031_loss-1.6706_accuracy-0.5280_val_loss-1.9804_val_accuracy-0.5219')
        model.load_weights(
            model_dir + '/AWA2/img/none/random_crop/ckpt-epoch0118_loss-0.3647_accuracy-0.8892_val_loss-2359.7251_val_accuracy-0.7427')
        print('check point found.')
    except Exception as e:
        print(e)
        print('no check point found.')
        raise RuntimeError
    # Evaluate
    # score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
    # print(score)
    total_count = 0
    positive_count = 0
    # the_sheet = np.zeros((class_num, class_num))
    for step in range(STEP_SIZE_VALID):
        batch_instance = next(val_data_gen)
        batch_predict = model.predict(batch_instance[0], batch_size=val_batch_size*10)
        _ = gc.collect()
        if batch_instance[0].shape[0] % 10 == 0:
            batch_instance_num = int(batch_instance[0].shape[0] / 10)
        else:
            raise RuntimeError
        for instance_No in range(batch_instance_num):
            instance_predict = batch_predict[instance_No * 10:instance_No * 10 + 10:, ...]
            instance_predict = instance_predict.sum(axis=0)
            maxarg = instance_predict.argmax(axis=0)
            # gt = -1
            # for i in range(len(batch_instance[1][instance_No * 10])):
            #     if batch_instance[1][instance_No * 10][i] == 1:
            #         if gt == -1:
            #             gt = i
            #         else:
            #             raise RuntimeError
            # the_sheet[gt][maxarg] += 1
            if batch_instance[1][instance_No * 10][maxarg] == 1:
                positive_count = positive_count + 1
            total_count = total_count + 1
        print('{}/{}-{}'.format(step, STEP_SIZE_VALID, positive_count / total_count))
    print(positive_count / total_count)
    a = 0
