import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.optimizers import SGD
import os, sys
import scipy.io as sio
from sklearn.cluster import SpectralClustering

import tensorflow.keras
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')

        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')

            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


class Scaler(keras.layers.Layer):
    def __init__(self, tau=0.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.tau = tau

    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='zeros'
        )

    def call(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * K.sigmoid(self.scale)

        else:
            scale = (1 - self.tau) * K.sigmoid(-self.scale)

        return inputs * K.sqrt(scale)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(keras.layers.Layer):
    def __init__(self, latent_dim=128, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), seed=42)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_config(self):
        base_config = super(Sampling, self).get_config()
        config = {'latent_dim': self.latent_dim}
        return dict(list(base_config.items()) + list(config.items()))

###########change dataset here##########
dataset = 'AWA2'
########################################

batch_size = 128
epochs = 1000

train_dir = './data/{}/IMG/train'.format(dataset)
val_dir = './data/{}/IMG/val'.format(dataset)
test_dir = './data/{}/IMG/test'.format(dataset)
image_size = 224
# attr_type = 'b','c','cmm','cms'
attr_type = 'cms'
feature_type = 'ft'

classname = pd.read_csv('./data/{}/classes.txt'.format(dataset), header=None, sep='\t')

if dataset == 'SUN':
    class_attr_shape = (102, )
    class_attr_dim = 102
    class_num = 717
    seen_class_num = 645
    unseen_class_num = 72
elif dataset == 'CUB':
    class_attr_shape = (312, )
    class_attr_dim = 312
    class_num = 200
    seen_class_num = 150
    unseen_class_num = 50
elif dataset == 'AWA2':
    class_attr_shape = (85, )
    class_attr_dim = 85
    class_num = 50
    seen_class_num = 40
    unseen_class_num = 10
elif dataset == 'plant':
    class_attr_shape = (46, )
    class_attr_dim = 46
    class_num = 38
    seen_class_num = 25
    unseen_class_num = 13

data_train = np.load('./data/{}/feature_label_attr/train/train_feature_{}.npy'.format(dataset,feature_type))
attr_train = np.load('./data/{}/feature_label_attr/train/train_attr_{}.npy'.format(dataset,attr_type))
label_train = np.load('./data/{}/feature_label_attr/train/train_label.npy'.format(dataset))

data_val = np.load('./data/{}/feature_label_attr/val/val_feature_{}.npy'.format(dataset,feature_type))
attr_val = np.load('./data/{}/feature_label_attr/val/val_attr_{}.npy'.format(dataset,attr_type))
label_val = np.load('./data/{}/feature_label_attr/val/val_label.npy'.format(dataset))

###########Model###########
x_inputs = Input(shape=(2048, ))
x = x_inputs
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
# z_mean = Dense(class_attr_dim, activation='relu')(x)
# z_var = Dense(class_attr_dim, activation='relu')(x)
z_mean = Dense(class_attr_dim)(x)
z_var = Dense(class_attr_dim)(x)
y_inputs = Input(shape=class_attr_shape)

###########Sampling###########
scaler = Scaler()
z_mean = scaler(z_mean, mode='positive')
z_var = scaler(z_var, mode='negative')
sampling = Sampling(class_attr_dim)
z = sampling([z_mean,z_var])

###########Decoder###########
ce_inputs = Input(shape=class_attr_shape)
x = ce_inputs
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs= Dense(2048, activation='relu')(x)

###########Modeling###########
encoder = Model(x_inputs, z)
decoder = Model(ce_inputs, outputs)
x_out = decoder(z)

vae = Model(inputs=[x_inputs, y_inputs],outputs=[x_out])

###########Model Loss Function###########
# xent_loss是重構loss
xent_loss = 0.5 * K.sum(K.mean((x_inputs - x_out)**2))

# K.square(z_mean - y) 為latent v ector 向每個class的均值看齊
kl_loss = - 0.5 * K.sum(1 + z_var - K.square(z_mean - y_inputs) - K.exp(z_var), axis=-1)

vae_loss = K.mean(xent_loss +  kl_loss)

###########Start train###########
vae.add_loss(vae_loss)
#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# vae.compile(optimizer='rmsprop')

vae.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))

# vae.summary()

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5)


vae.fit(
    [data_train, attr_train],
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([data_val, attr_val], None),
    callbacks=[history,early_stopping,learning_rate_reduction]
)
encoder.save('./model/{}/encoder_{}_{}.h5'.format(dataset,feature_type,attr_type))

###########make the attr.mat file###########
image_gen = ImageDataGenerator()
train_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    color_mode="rgb",
    target_size=(image_size, image_size),
    class_mode='sparse',
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

