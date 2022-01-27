import numpy as np
import matplotlib.pyplot as plt
import class_name_index
import configure
import math


class_name_map = class_name_index.ClassNameIndexMap()
features = np.load(configure.ckpt_dir + '/data/features.npy')
labels = np.load(configure.ckpt_dir + '/data/labels.npy')
# class_features = []
# # 不同class分開
# for class_No in range(configure.class_num):
#     class_features.append(features[np.where(labels == class_No)[0], ...])

# # 各class算距離
# class_distance_matrix = []
# for i in range(len(class_features)):
#     class_feature = class_features[i]
#     distance_matrix = np.zeros((class_feature.shape[0], class_feature.shape[0]))
#     for feature_index in range(class_feature.shape[0]):
#         target = class_feature[feature_index, ...]
#         distance_matrix[feature_index, ...] = np.power(np.sum(np.power(class_feature - target, 2), axis=1), 0.5)
#         print('{}-{}'.format(i, feature_index))
#     class_distance_matrix.append(distance_matrix)
# for i in range(len(class_distance_matrix)):
#     np.save(configure.ckpt_dir + '/data/{}_distance_matrix.npy'.format(i), class_distance_matrix[i])

# 全部算距離
feature_num = features.shape[0]
for class_No in range(configure.class_num):
    class_member_index = np.where(labels == class_No)[0]
    class_member_num = class_member_index.shape[0]
    class_distance_matrix = np.zeros((class_member_num, feature_num), dtype=float)
    for class_member_No in range(class_member_num):
        class_distance_matrix[class_member_No, ...] = np.power(np.sum(np.power(features - features[class_member_index[class_member_No], ...], 2), axis=1), 0.5)
        print('{}-{}/{}'.format(class_No, class_member_No, class_member_num))
    np.save(configure.ckpt_dir + '/data/class{}_all_distance_matrix.npy'.format(class_No), class_distance_matrix)

# class_distance_matrix = []
# for i in range(configure.class_num):
#     class_distance_matrix.append(np.load(configure.ckpt_dir + 'data/{}_distance_matrix.npy'.format(i)))
# class_distance_mean = []
# class_distance_std = []
# grid_num = 11
# for i in range(configure.class_num):
#     class_distance_mean.append(np.average(class_distance_matrix[i], axis=1))
#     class_distance_std.append(np.std(class_distance_matrix[i], axis=1))
# mean_of_mean = []
# std_of_mean = []
# mean_of_std = []
# std_of_std = []
# for i in range(configure.class_num):
#     mean_of_mean.append(np.average(class_distance_mean[i], axis=0))
#     std_of_mean.append(np.std(class_distance_mean[i], axis=0))
#     mean_of_std.append(np.average(class_distance_std[i], axis=0))
#     std_of_std.append(np.std(class_distance_std[i], axis=0))
# plt.barh(y=class_name_map.name_list(), width=mean_of_mean, height=0.5, )
# plt.title('mean_of_mean')
# plt.show()
# plt.barh(y=class_name_map.name_list(), width=std_of_mean, height=0.5, )
# plt.title('std_of_mean')
# plt.show()
# plt.barh(y=class_name_map.name_list(), width=mean_of_std, height=0.5, )
# plt.title('mean_of_std')
# plt.show()
# plt.barh(y=class_name_map.name_list(), width=std_of_std, height=0.5, )
# plt.title('std_of_std')
# plt.show()
a = 0

