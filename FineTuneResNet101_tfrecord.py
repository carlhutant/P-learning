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
dataset = 'imagenet'
# datatype: img, tfrecord
datatype = 'tfrecord'
# data preprocess: color_diff_121, none
preprocess = 'none'
##############################

batch_size = 128
dataset_directory = 'F:/Dataset/{}/{}/{}/'.format(dataset, datatype, preprocess)
IMG_SHAPE = 224
train_cardinality = 0
val_cardinality = 0
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
elif dataset == 'imagenet':
    class_attr_shape = (85,)
    class_attr_dim = 85
    class_num = 1000
    seen_class_num = 1000
    unseen_class_num = 10
    train_cardinality = 1281167
    val_cardinality = 50000
elif dataset == 'plant':
    class_attr_shape = (46,)
    class_attr_dim = 46
    class_num = 38
    seen_class_num = 25
    unseen_class_num = 13

if preprocess == 'color_diff_121':
    IMG_channel = 6
elif preprocess == 'none':
    IMG_channel = 3
#
# image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
# train_data_gen = image_gen.flow_from_directory(
#     batch_size=batch_size,
#     directory=train_dir,
#     shuffle=True,
#     color_mode="rgb",
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     seed=42
# )
#
# image_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
#
# val_data_gen = image_gen_val.flow_from_directory(
#     batch_size=batch_size,
#     directory=val_dir,
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     color_mode="rgb",
#     seed=42
# )
#
# class_weights = class_weight.compute_class_weight(
#            'balanced',
#             np.unique(train_data_gen.classes),
#             train_data_gen.classes)


# 1 tfrecord dataset
def tf_parse1(raw_example):
    parsed = tf.train.Example.FromString(raw_example.numpy())
    feature = np.array(parsed.features.feature['feature'].float_list.value)
    label = np.array(parsed.features.feature['label'].int64_list.value)
    return feature, label


# 2 tfrecord dataset
def tf_parse2(raw_example):
    example = tf.io.parse_example(
        raw_example[tf.newaxis], {
            'feature': tf.io.FixedLenFeature(shape=(1, IMG_SHAPE*IMG_SHAPE*IMG_channel), dtype=tf.float32),
            'label': tf.io.FixedLenFeature(shape=seen_class_num, dtype=tf.float32)
        })
    feature = tf.reshape(example['feature'][0], [IMG_SHAPE, IMG_SHAPE, IMG_channel])
    # feature = preprocess_input(feature)
    label = example['label'][0]
    return feature, label


train_files_list = tf.data.Dataset.list_files(dataset_directory + 'train.tfrecord*')
val_files_list = tf.data.Dataset.list_files(dataset_directory + 'val.tfrecord*')
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

train_dataset = train_dataset.map(tf_parse2)
val_dataset = val_dataset.map(tf_parse2)
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
# train_dataset = train_dataset.repeat(15)
# val_dataset = val_dataset.repeat(15)
# val_dataset.shuffle()
# x = train_dataset.take(1)
# a = 0
# # Fine tune or Retrain ResNet101
model = ResNet101(weights='imagenet', include_top=True)

# # lock the model
# for layer in base_model.layers:
#     layer.trainable = False

# add a global averge pollinf layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)

# add a dense
# x = Dense(2048, activation='relu')(x)

# add a classifier
# predictions = Dense(seen_class_num, activation='softmax')(x)

# Constructure
# model = Model(inputs=base_model.input, outputs=predictions)

# compile
# model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#               , loss='categorical_crossentropy',metrics=['accuracy'])
# keras.utils.plot_model(model, to_file='model2.png', show_shapes=True, show_dtype=True, show_layer_names=True,
#                            rankdir="TB", expand_nested=False, dpi=96, )  # 儲存模型圖

model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
              , loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_cardinality // batch_size
STEP_SIZE_VALID = val_cardinality // batch_size

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# model.save('./model/{}/none_finetune_tfrecord/ResNet101_none_step0_epoch{}.h5'.format(dataset, 0))
epochs = 15
for i in range(epochs):
    print("step 1 epoch {}:".format(i+1))
    model.fit(train_dataset, epochs=1, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=val_dataset,
              validation_steps=STEP_SIZE_VALID, callbacks=[early_stopping])
    # model.save('./model/{}/none_finetune_tfrecord/ResNet101_none_step1_epoch{}.h5'.format(dataset, i))

for layer in model.layers[:335]:
    layer.trainable = False
for layer in model.layers[335:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_cardinality // batch_size
STEP_SIZE_VALID = val_cardinality // batch_size

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
epochs = 10
print("step 2:")
for i in range(epochs):
    print("step 2 epoch {}:".format(i+1))
    model.fit(train_dataset, epochs=1, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=val_dataset,
              validation_steps=STEP_SIZE_VALID, callbacks=[early_stopping])
    # model.save('./model/{}/none_finetune_tfrecord/ResNet101_none_step2_epoch{}.h5'.format(dataset, i))

# ## Evaluate
# model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#               , loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights("./model/AWA2/FineTuneResNet101_edge_with_head.h5")
# STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size
# score = model.evaluate_generator(generator=val_data_gen, steps=STEP_SIZE_VALID)
# print(score)
