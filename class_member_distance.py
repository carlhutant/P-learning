import numpy as np
import class_name_index
import configure


class_name_map = class_name_index.ClassNameIndexMap()
features = np.load(configure.ckpt_dir + '/data/features.npy')
labels = np.load(configure.ckpt_dir + '/data/labels.npy')
a = 0

