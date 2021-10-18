import os
from configure import *
from pathlib import Path
target = Path('E:/Model/AWA2/npy/color_diff_121_abs/random_crop/')

walk_generator = os.walk(target)
_, _, files = next(walk_generator)
for file in files:
    if file.startswith('ckpt-epoch') and file.endswith('.index'):
        print(file)
a = 0
