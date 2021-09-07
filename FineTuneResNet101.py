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

# Import data
# change the dataset here###
dataset = 'AWA2'
##############################

batch_size = 16
# train_dir = './data/{}/IMG/train'.format(dataset)
train_dir = './data/{}/test_draw'.format(dataset)

# extend: [1.0499614477157593, 0.8599420189857483] edge: [0.7108655571937561, 0.8704637289047241]   # 訓練環境
# val_dir = './data/{}/IMG/val'.format(dataset) # 辨識資料

# extend: [18.021373748779297, 0.07749495655298233] edge: [13.31424617767334, 0.03162802383303642]   # 訓練環境
val_dir = './data/{}/IMG/new_val_b_y'.format(dataset)   # 辨識資料

# extend: [3.345625877380371, 0.5540574789047241] edge: [2.9107425212860107, 0.5270287394523621]   # 訓練環境
# val_dir = './data/{}/IMG/val_edge'.format(dataset)    # 辨識資料
IMG_SHAPE = 224

epochs = 15

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

image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
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

image_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='categorical',
    color_mode="rgb",
    seed=42
)

class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_data_gen.classes),
            train_data_gen.classes)


## Fine tune or Retrain ResNet101
base_model = ResNet101(weights='imagenet', include_top=True)

# # lock the model
# for layer in base_model.layers:
#     layer.trainable = False

# add a global averge pollinf layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a dense
x = Dense(1024, activation='relu')(x)

# add a classifier
predictions = Dense(seen_class_num, activation='softmax')(x)

# Constructure
model = Model(inputs=base_model.input, outputs=predictions)


# compile
# model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#               , loss='categorical_crossentropy',metrics=['accuracy'])
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                            rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖

model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

STEP_SIZE_TRAIN = train_data_gen.n // train_data_gen.batch_size
STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size

model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    #                     class_weight=class_weights,
                    callbacks=[early_stopping]
                    )

epochs = 10

for layer in model.layers[:335]:
    layer.trainable = False
for layer in model.layers[335:]:
    layer.trainable = True

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

STEP_SIZE_TRAIN = train_data_gen.n // train_data_gen.batch_size
STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size

model.fit_generator(train_data_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=STEP_SIZE_VALID,
                    #                     class_weight=class_weights,
                    callbacks=[early_stopping]
                    )

new_model = Model(model.inputs, model.layers[-3].output)

new_model.summary()

new_model.save('./model/{}/FineTuneResNet101.h5'.format(dataset))

# ## Evaluate
model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights("./model/AWA2/FineTuneResNet101_edge_with_head.h5")
# STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size
score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
print(score)
