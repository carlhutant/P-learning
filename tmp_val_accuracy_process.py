import os
from configure import *
from pathlib import Path
target = Path('G:/carl/p_learning/model/AWA2/multi_domain/img_img/none_color_diff_121_abs_3ch/random_crop')

walk_generator = os.walk(target)
_, _, files = next(walk_generator)
files.sort()
for file in files:
    if file.startswith('ckpt-epoch') and file.endswith('.index'):
        print(file)
a = 0
