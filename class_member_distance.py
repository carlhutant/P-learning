import numpy as np
import class_name_index
import configure
import math


class_name_map = class_name_index.ClassNameIndexMap()
features = np.load(configure.ckpt_dir + '/data/features.npy')
labels = np.load(configure.ckpt_dir + '/data/labels.npy')
class_features = []
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
distance_matrix = np.zeros((feature_num, feature_num), dtype=float)
for feature_index in range(feature_num):
    distance_matrix[feature_index, ...] = np.power(np.sum(np.power(features - features[feature_index, ...], 2), axis=1),
                                                   0.5)
    print('{}/{}'.format(feature_index, feature_num))
np.save(configure.ckpt_dir + '/data/all_distance_matrix.npy', distance_matrix)
# class_distance_matrix = []
#
# for i in range(configure.class_num):
#     class_distance_matrix.append(np.load(configure.ckpt_dir + 'data/{}_distance_matrix.npy'.format(i)))
# class_grid_count = []
# grid_num = 11
# for i in range(configure.class_num):
#     grid = math.ceil(class_distance_matrix[i].max() / 10)
#     for member_index in range(class_distance_matrix[i].shape[0]):
#         grid_count = np.empty(grid_num, dtype=np.int)
#         for start in range(11):
#             grid_count[i] = class_distance_matrix[i][member_index][np.where(np.logical_and(class_distance_matrix[i][member_index] >= start*grid, class_distance_matrix[i][member_index] < start*grid + grid))].shape[0]
#
# a = 0

