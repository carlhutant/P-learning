import os
from configure import *
from pathlib import Path
target = Path(model_dir + '/AWA2/tfrecord/color_diff_121_abs_3ch/random_crop/')

walk_generator = os.walk(target)
_, dirs, files = next(walk_generator)
files.sort()
dirs.sort()
# for file in files:
#     if file.startswith('ckpt-epoch') and file.endswith('.index'):
#         print(file)
for dir in dirs:
    print(dir)
a = 0
