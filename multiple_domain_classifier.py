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

if GPU_memory_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print('building two resnet model...')
base_model = ResNet101(weights=None, include_top=False, input_shape=(224, 224, channel))
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(class_num, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=False),
              loss='categorical_crossentropy', metrics=['accuracy'])
try:
    model.load_weights(
        Path('E:/Model/AWA2/img/none/ckpt-epoch0140_loss-0.3118_accuracy-0.9060_val_loss-1758.8971_val_accuracy-0.7381')
    )
    print('check point found.')
except Exception as e:
    print(e)
    raise RuntimeError
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                        rankdir="TB", expand_nested=False, dpi=96, )
model = Model(inputs=model.input, outputs=model.layers[-2].output)

base_model2 = ResNet101(weights=None, include_top=False, input_shape=(224, 224, channel2))
x2 = GlobalAveragePooling2D()(base_model2.output)
predictions2 = Dense(class_num, activation='softmax')(x2)
model2 = Model(inputs=base_model2.input, outputs=predictions2)
model2.compile(optimizer=SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=False),
               loss='categorical_crossentropy', metrics=['accuracy'])
try:
    model2.load_weights(Path('E:/Model/AWA2/img/none/diff'))
    print('check point found.')
except Exception as e:
    print(e)
    raise RuntimeError
# keras.utils.plot_model(model2, to_file='model2.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                        rankdir="TB", expand_nested=False, dpi=96, )
model2 = Model(inputs=model2.input, outputs=model2.layers[-2].output)
print('done.')

if datatype2 is None:
    raise RuntimeError
else:
    train_data_gen = domains_feature_generator(
        target_directory=train_dir,
        target2_directory=train_dir2,
        model=model,
        model2=model2,
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
    val_data_gen = domains_feature_generator(
        target_directory=val_dir,
        target2_directory=val_dir2,
        model=model,
        model2=model2,
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

next(train_data_gen)
next(val_data_gen)

# while True:
#     a = next(train_data_gen)
#     print('a')

if multi_GPU:
    raise RuntimeError
else:
    input_layer = tf.keras.Input(shape=(4096,))
    predictions = Dense(class_num, activation='softmax')(input_layer)
    # Construction
    model = Model(inputs=input_layer, outputs=predictions)

# keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                        rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖


early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(ckpt_dir + 'ckpt-epoch{epoch:04d}'
                                   + '_loss-{loss:.4f}'
                                   + '_accuracy-{accuracy:.4f}'
                                   + '_val_loss-{val_loss:.4f}'
                                   + '_val_accuracy-{val_accuracy:.4f}',
                                   save_weights_only=True,
                                   save_freq='epoch',
                                   verbose=0)
reduce_LR_on_plateau = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=10,
                                         verbose=1,
                                         min_delta=1,
                                         min_lr=0.00001)

STEP_SIZE_TRAIN = math.ceil(train_cardinality / train_batch_size)
STEP_SIZE_VALID = math.ceil(val_cardinality / val_batch_size)

epochs = 400

model.compile(optimizer=SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=False)
              , loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[model_checkpoint]
                    )

