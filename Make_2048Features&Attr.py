import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os, sys

import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet import ResNet101,preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf

###########change dataset here##########
dataset = 'AWA2'
########################################

batch_size = 1
image_size = 224
train_dir = './data/{}/IMG/train'.format(dataset)
val_dir = './data/{}/IMG/val'.format(dataset)
test_dir = './data/{}/IMG/test'.format(dataset)
attr_binary_path = './data/{}/predicate-matrix-binary.txt'.format(dataset)
attr_continous_path = './data/{}/predicate-matrix-continuous.txt'.format(dataset)


classname = pd.read_csv('./data/{}/classes.txt'.format(dataset), header=None, sep='\t')

############Load Model############
model_ft = load_model('./model/{}/FineTuneResNet101.h5'.format(dataset))
# model_rt = load_model('./model/{}/RetrainResNet101.h5'.format(dataset))

############Generator############
image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    color_mode="rgb",
    target_size=(image_size, image_size),
    class_mode='sparse',
    shuffle=False,
    seed=42
)

val_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    target_size=(image_size, image_size),
    class_mode='sparse',
    color_mode="rgb",
    seed=42

)

test_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    target_size=(image_size, image_size),
    class_mode='sparse',
    color_mode="rgb",
    seed=42
)
train_gen.classes

############Binary attr############
attr_list_b = []
RealCE_binary = pd.read_csv(attr_binary_path,header=None,sep='\t')
for idx in range(len(RealCE_binary)):
    tmp = RealCE_binary[0][idx].split(' ')
    attr = [float(i) for i in tmp if i!='']
    attr = np.array(attr)
    attr_list_b.append(attr)

############Continous############
attr_list_c = []
RealCE_continous = pd.read_csv(attr_continous_path,header=None,sep = '\t')
for idx in range(len(RealCE_continous)):
    tmp = RealCE_continous[0][idx].split(' ')
    attr = [float(i) for i in tmp if i!='']
    attr = np.array(attr)
    attr_list_c.append(attr)

############Continous attr min max############
attr_list_cmm = []
RealCE_continous = pd.read_csv(attr_continous_path,header=None,sep = '\t')
for idx in range(len(RealCE_continous)):
    tmp = RealCE_continous[0][idx].split(' ')
    attr = [float(i) for i in tmp if i!='']
    attr = np.array(attr)
    attr = (attr - np.min(attr)) / (np.max(attr) - np.min(attr))
    attr_list_cmm.append(attr)

############Continous attr mean std############
attr_list_cms = []
RealCE_continous = pd.read_csv(attr_continous_path,header=None,sep = '\t')
for idx in range(len(RealCE_continous)):
    tmp = RealCE_continous[0][idx].split(' ')
    attr = [float(i) for i in tmp if i!='']
    attr = np.array(attr)
    attr = (attr - np.mean(attr)) / np.std(attr)
    attr_list_cms.append(attr)

############make the convert_attr_list for train,val############
train_attr_list_b = []
train_attr_list_c = []
train_attr_list_cmm = []
train_attr_list_cms = []

for k ,v in train_gen.class_indices.items():
    idx = np.where(classname[1] == k)
    train_attr_list_b.append(attr_list_b[idx[0][0]])
    train_attr_list_c.append(attr_list_c[idx[0][0]])
    train_attr_list_cmm.append(attr_list_cmm[idx[0][0]])
    train_attr_list_cms.append(attr_list_cms[idx[0][0]])

############make the convert_attr_list for test############
test_attr_list_b = []
test_attr_list_c = []
test_attr_list_cmm = []
test_attr_list_cms = []

for k ,v in test_gen.class_indices.items():
    idx = np.where(classname[1] == k)
    test_attr_list_b.append(attr_list_b[idx[0][0]])
    test_attr_list_c.append(attr_list_c[idx[0][0]])
    test_attr_list_cmm.append(attr_list_cmm[idx[0][0]])
    test_attr_list_cms.append(attr_list_cms[idx[0][0]])

############train data############
count = 0
ft_feature = np.array([], dtype='float32').reshape(0, 2048)
rt_feature = np.array([], dtype='float32').reshape(0, 2048)

attr_b = []
attr_c = []
attr_cmm = []
attr_cms = []
label_list = []
#while count < train_gen.n:
while count < 10:
    print("train {}/{}".format(count, train_gen.n))
    data, label = train_gen.next()
    # fine tune feature
    after_predict = model_ft.predict(data)
    ft_feature = np.concatenate((ft_feature, after_predict))

    # retrain feature
    #     after_predict = model_rt.predict(data)
    #     rt_feature = np.concatenate((rt_feature,after_predict))

    # attr
    for l in label:
        attr_b.append(train_attr_list_b[int(l)])
        attr_c.append(train_attr_list_c[int(l)])
        attr_cmm.append(train_attr_list_cmm[int(l)])
        attr_cms.append(train_attr_list_cms[int(l)])

        # label
        label_list.append(int(l))

    count += len(data)

attr_b = np.array(attr_b)
attr_c = np.array(attr_c)
attr_cmm = np.array(attr_cmm)
attr_cms = np.array(attr_cms)
label_list = np.array(label_list)

print(ft_feature.shape)
print(rt_feature.shape)
print(attr_b.shape)
print(attr_c.shape)
print(attr_cmm.shape)
print(attr_cms.shape)
print(label_list.shape)

np.save('./data/{}/feature_label_attr/train/train_feature_ft.npy'.format(dataset),ft_feature)
np.save('./data/{}/feature_label_attr/train/train_feature_rt.npy'.format(dataset),rt_feature)

np.save('./data/{}/feature_label_attr/train/train_attr_b.npy'.format(dataset),attr_b)
np.save('./data/{}/feature_label_attr/train/train_attr_c.npy'.format(dataset),attr_c)
np.save('./data/{}/feature_label_attr/train/train_attr_cmm.npy'.format(dataset),attr_cmm)
np.save('./data/{}/feature_label_attr/train/train_attr_cms.npy'.format(dataset),attr_cms)

np.save('./data/{}/feature_label_attr/train/train_label.npy'.format(dataset),label_list)

############val data############
'''count = 0
ft_feature = np.array([], dtype='float32').reshape(0, 7, 7, 2048)
rt_feature = np.array([], dtype='float32').reshape(0, 7, 7, 2048)

attr_b = []
attr_c = []
attr_cmm = []
attr_cms = []
label_list = []
#while count < val_gen.n:
while count < 1024:
    print("val {}/{}".format(count, train_gen.n))
    data, label = val_gen.next()
    # fine tune feature
    after_predict = model_ft.predict(data)
    ft_feature = np.concatenate((ft_feature, after_predict))

    # retrain feature
    #     after_predict = model_rt.predict(data)
    #     rt_feature = np.concatenate((rt_feature,after_predict))

    # attr
    for l in label:
        attr_b.append(train_attr_list_b[int(l)])
        attr_c.append(train_attr_list_c[int(l)])
        attr_cmm.append(train_attr_list_cmm[int(l)])
        attr_cms.append(train_attr_list_cms[int(l)])

        # label
        label_list.append(int(l))

    count += len(data)

attr_b = np.array(attr_b)
attr_c = np.array(attr_c)
attr_cmm = np.array(attr_cmm)
attr_cms = np.array(attr_cms)
label_list = np.array(label_list)

print(ft_feature.shape)
print(rt_feature.shape)
print(attr_b.shape)
print(attr_c.shape)
print(attr_cmm.shape)
print(attr_cms.shape)
print(label_list.shape)

np.save('./data/{}/feature_label_attr/val/val_feature_ft.npy'.format(dataset),ft_feature)
np.save('./data/{}/feature_label_attr/val/val_feature_rt.npy'.format(dataset),rt_feature)

np.save('./data/{}/feature_label_attr/val/val_attr_b.npy'.format(dataset),attr_b)
np.save('./data/{}/feature_label_attr/val/val_attr_c.npy'.format(dataset),attr_c)
np.save('./data/{}/feature_label_attr/val/val_attr_cmm.npy'.format(dataset),attr_cmm)
np.save('./data/{}/feature_label_attr/val/val_attr_cms.npy'.format(dataset),attr_cms)

np.save('./data/{}/feature_label_attr/val/val_label.npy'.format(dataset),label_list)

############test data############
count = 0
ft_feature = np.array([], dtype='float32').reshape(0, 7, 7, 2048)
rt_feature = np.array([], dtype='float32').reshape(0, 7, 7, 2048)

attr_b = []
attr_c = []
attr_cmm = []
attr_cms = []
label_list = []
#while count < test_gen.n:
while count < 1024:
    print("test {}/{}".format(count, train_gen.n))
    data, label = test_gen.next()
    # fine tune feature
    after_predict = model_ft.predict(data)
    ft_feature = np.concatenate((ft_feature, after_predict))

    # retrain feature
    #     after_predict = model_rt.predict(data)
    #     rt_feature = np.concatenate((rt_feature,after_predict))

    # attr
    for l in label:
        attr_b.append(test_attr_list_b[int(l)])
        attr_c.append(test_attr_list_c[int(l)])
        attr_cmm.append(test_attr_list_cmm[int(l)])
        attr_cms.append(test_attr_list_cms[int(l)])

        # label
        label_list.append(int(l))

    count += len(data)

attr_b = np.array(attr_b)
attr_c = np.array(attr_c)
attr_cmm = np.array(attr_cmm)
attr_cms = np.array(attr_cms)
label_list = np.array(label_list)

print(ft_feature.shape)
print(rt_feature.shape)
print(attr_b.shape)
print(attr_c.shape)
print(attr_cmm.shape)
print(attr_cms.shape)
print(label_list.shape)

np.save('./data/{}/feature_label_attr/test/test_feature_ft.npy'.format(dataset),ft_feature)
np.save('./data/{}/feature_label_attr/test/test_feature_rt.npy'.format(dataset),rt_feature)

np.save('./data/{}/feature_label_attr/test/test_attr_b.npy'.format(dataset),attr_b)
np.save('./data/{}/feature_label_attr/test/test_attr_c.npy'.format(dataset),attr_c)
np.save('./data/{}/feature_label_attr/test/test_attr_cmm.npy'.format(dataset),attr_cmm)
np.save('./data/{}/feature_label_attr/test/test_attr_cms.npy'.format(dataset),attr_cms)

np.save('./data/{}/feature_label_attr/test/test_label.npy'.format(dataset),label_list)'''