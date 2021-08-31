import numpy as np
from pandas import read_csv

from keras import backend as K
from keras.layers import Layer
from keras.models import load_model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Scaler(Layer):
    def __init__(self, tau=.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.tau = tau

    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='zeros'
        )

    def call(self, inputs, mode='positive'):
        return inputs * K.sqrt((self.tau + (1 - self.tau) * K.sigmoid(self.scale))
                                   if mode == 'positive' else (1 - self.tau) * K.sigmoid(-self.scale))

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(Layer):
    def __init__(self, latent_dim=128, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], self.latent_dim), seed=42)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_config(self):
        base_config = super(Sampling, self).get_config()
        config = {'latent_dim': self.latent_dim}
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = { 'Scaler': Scaler, 'Sampling': Sampling }
encoder = load_model('./model/AWA2/encoder_ft_cms.h5', custom_objects, compile=False)

dataset = 'AWA2'

if dataset == 'AWA2':
    class_num = 50
    seen_class_num = 40
    unseen_class_num = 10

################Seen################
data_train = np.load(
    f'./data/{dataset}/feature_label_attr/train/train_feature_ft.npy')
label_train = np.load(
    f'./data/{dataset}/feature_label_attr/train/train_label.npy')

seen_predict = encoder.predict(data_train)

seen_attr = [[] for _ in range(seen_class_num)]
count_class = [0] * seen_class_num
for idx in range(len(seen_predict)):
    l = label_train[idx]
    if len(seen_attr[l]):
        seen_attr[l] += seen_predict[idx]
    else:
        seen_attr[l] = seen_predict[idx]

    count_class[l] += 1

for i in range(seen_class_num):
    seen_attr[i] = seen_attr[i] / count_class[i]

seen_attr = np.array(seen_attr)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(seen_attr, range(0, 40))
predict_label = neigh.predict(seen_predict)
print(len([i for i in range(len(seen_predict)) if predict_label[i] == label_train[i]]) / len(seen_predict))

################Unseen################
data_test = np.load(
    f'./data/{dataset}/feature_label_attr/test/test_feature_ft.npy')
label_test = np.load(
    f'./data/{dataset}/feature_label_attr/test/test_label.npy')

unseen_predict = encoder.predict(data_test)

unseen_attr = [[] for _ in range(unseen_class_num)]
count_class = [0] * unseen_class_num
for idx in range(len(unseen_predict)):
    l = label_test[idx]
    if len(unseen_attr[l]):
        unseen_attr[l] += unseen_predict[idx]
    else:
        unseen_attr[l] = unseen_predict[idx]

    count_class[l] += 1

for i in range(unseen_class_num):
    unseen_attr[i] = unseen_attr[i] / count_class[i]

unseen_attr = np.array(unseen_attr)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(unseen_attr, range(0, 10))
predict_label = neigh.predict(unseen_predict)
print(len([i for i in range(len(unseen_predict)) if predict_label[i] == label_test[i]]) / len(unseen_predict))
