import warnings
import os
import tensorflow as tf
import numpy as np
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
from data_generator import *
from configure import *

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    if datatype2 is None:
        train_data_gen = parallel_data_generator(
            target_directory=train_dir,
            batch_size=train_batch_size,
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
        val_data_gen = parallel_data_generator(
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
    else:
        raise RuntimeError

    next(train_data_gen)
    next(val_data_gen)

    # while True:
    #     a = next(train_data_gen)
    #     print('a')
    # test final_batch_opt
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

    if GPU_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Fine tune or Retrain ResNet101
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

    # STEP_SIZE_TRAIN = 1
    STEP_SIZE_TRAIN = math.ceil(train_cardinality / train_batch_size)
    STEP_SIZE_VALID = math.ceil(val_cardinality / val_batch_size)

    # try:
    #     # model = tf.keras.models.load_model(ckp_path)
    #     # model = tf.keras.models.load_model('D:\\Download\\P_learning\\model\\AWA2\img\\none\\random_crop\\ckpt-epoch0001_loss-1.6212_accuracy-0.5446_val_loss-3.0151_val_accuracy-0.5061')
    #     # model.load_weights(ckp_path)
    #     # model.load_weights('D:\\Download\\P_learning\\model\\AWA2\img\\none\\random_crop\\ckpt-epoch0031_loss-1.6706_accuracy-0.5280_val_loss-1.9804_val_accuracy-0.5219')
    #     model.load_weights(model_dir + '/AWA2/img/none/random_crop/ckpt-epoch0118_loss-0.3647_accuracy-0.8892_val_loss-2359.7251_val_accuracy-0.7427')
    #     print('check point found.')
    # except Exception as e:
    #     print(e)
    #     print('no check point found.')

    epochs = 200

    # a = model.layers[-1].weights[0][0]
    # for layer in model.layers:
    #     layer.trainable = False
    # for layer in model.layers[-1:]:
    #     layer.trainable = True
    # b = model.layers[-1].weights[0][0]
    model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.5, nesterov=False), loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # c = model.layers[-1].weights[0][0]
    # model.optimizer.learning_rate.assign(0)
    model.fit_generator(train_data_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=epochs,
                        # validation_data=val_data_gen,
                        validation_steps=STEP_SIZE_VALID,
                        # callbacks=[model_checkpoint]
                        )
    train_data_gen.send(1)
    val_data_gen.send(1)
    # d = model.layers[-1].weights[0][0]
    # model.optimizer.learning_rate.assign(0.001)
    # model.fit_generator(train_data_gen,
    #                     steps_per_epoch=5,
    #                     epochs=1,
    #                     validation_data=val_data_gen,
    #                     validation_steps=1,
    #                     # callbacks=[model_checkpoint]
    #                     )
    # e = model.layers[-1].weights[0][0]
    # f = 0
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

    # Evaluate
    # score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
    # print(score)
    # total_count = 0
    # positive_count = 0
    # for i in range(STEP_SIZE_VALID):
    #     print(i)
    #     x = next(val_data_gen)
    #     y = model.predict(x[0])
    #     y = y.sum(axis=0)
    #     maxarg = y.argmax(axis=0)
    #     if x[1][0][maxarg] == 1:
    #         positive_count = positive_count + 1
    #     total_count = total_count + 1
    # print(positive_count/total_count)
