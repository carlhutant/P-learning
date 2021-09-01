import os
import tarfile

directory_path = 'D:/Download/imagenet/ILSVRC2012_img_train/'
walk_generator = os.walk(directory_path)
files = next(walk_generator)[2]
for file in files:
    file_name, _ = os.path.splitext(file)
    os.mkdir(directory_path+file_name)
    tar = tarfile.open(directory_path+file, 'r')
    tar.extractall(directory_path+file_name)
    print(file_name+' done.')
