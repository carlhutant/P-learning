import os
import scipy.io as io
train_img_directory = 'D:/Download/imagenet/ILSVRC2012_img_train/'
val_img_directory = './data/imagenet/img/val/'
val_GT_file_path = './data/imagenet/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt '
meta_path = './data/imagenet/devkit/ILSVRC2012_devkit_t12/data/meta.mat'

mat = io.loadmat(meta_path)
synsets = mat['synsets']

walk_generator = os.walk(train_img_directory)
train_root, _, train_files = next(walk_generator)
walk_generator = os.walk(val_img_directory)
val_root, _, val_files = next(walk_generator)
f = open(val_GT_file_path, 'r')
label_list = []
txt = f.read()
txt = txt.split('\n')
for i in range(50000):
    label = synsets[int(txt[i]) - 1][0][1][0]
    save_dir = val_root + label + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    os.replace(val_root + val_files[i], save_dir + val_files[i])
a = 0
