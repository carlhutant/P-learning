import warnings

warnings.filterwarnings('ignore')

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

# Import data
# change the dataset here###
dataset = 'AWA2'
# datatype: img, tfrecord
datatype = 'img'
# data preprocess: color_diff_121, none
preprocess = 'none'
##############################

batch_size = 128
train_dir = 'E:/Dataset/{}/{}/train/'.format(dataset, datatype)
val_dir = 'E:/Dataset/{}/{}/val/'.format(dataset, datatype)
IMG_SHAPE = 224

epochs = 20

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
elif dataset == 'plant':
    class_attr_shape = (46,)
    class_attr_dim = 46
    class_num = 38
    seen_class_num = 25
    unseen_class_num = 13
elif dataset == 'imagenet':
    class_attr_shape = (0,)
    class_attr_dim = 0
    class_num = 1000
    seen_class_num = 1000
    unseen_class_num = 0

# image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
image_gen = ImageDataGenerator()
# image_gen = ImageDataGenerator()
train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=False,
    color_mode="rgb",
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='categorical',
    seed=42
)

# image_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
image_gen_val = ImageDataGenerator()
val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='categorical',
    color_mode="rgb",
    seed=42
)

# class_weights = class_weight.compute_class_weight(
#            'balanced',
#             np.unique(train_data_gen.classes),
#             train_data_gen.classes)
## Fine tune or Retrain ResNet101
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

model.compile(optimizer=SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

STEP_SIZE_TRAIN = train_data_gen.n // train_data_gen.batch_size
STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size

model.save('./model/{}/{}_{}/ResNet101_step0.h5'.format(dataset, preprocess, datatype))

model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    #                     class_weight=class_weights,
                    callbacks=[early_stopping]
                    )
model.save('./model/{}/{}_{}/ResNet101_lr01.h5'.format(dataset, preprocess, datatype))

epochs = 10
model.compile(optimizer=SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    #                     class_weight=class_weights,
                    callbacks=[early_stopping]
                    )
model.save('./model/{}/{}_{}/ResNet101_lr001.h5'.format(dataset, preprocess, datatype))

epochs = 10
model.compile(optimizer=SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    #                     class_weight=class_weights,
                    callbacks=[early_stopping]
                    )
model.save('./model/{}/{}_{}/ResNet101_lr0001.h5'.format(dataset, preprocess, datatype))

epochs = 10
model.compile(optimizer=SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    #                     class_weight=class_weights,
                    callbacks=[early_stopping]
                    )
model.save('./model/{}/{}_{}/ResNet101_lr00001.h5'.format(dataset, preprocess, datatype))

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
# model.save('./model/{}/{}_{}/ResNet101_step2.h5'.format(dataset, preprocess, datatype))

# # Evaluate
# model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#               , loss='categorical_crossentropy', metrics=['accuracy'])
# # model.load_weights("./model/AWA2/FineTuneResNet101_edge_with_head.h5")
# # STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size
# score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
# print(score)
