import os
import shutil
import configure
from pathlib import Path

target_dir = Path(configure.model_dir).joinpath('AWA2').joinpath('img').joinpath('none').joinpath('random_crop')
walk_generator = os.walk(target_dir)
root, directories, files = next(walk_generator)
for d in directories:
    if d.startswith('ckpt-epoch'):
        walk_generator2 = os.walk(Path(target_dir).joinpath(d).joinpath('variables'))
        root2, _, files2 = next(walk_generator2)
        for f in files2:
            shutil.copy(Path(root2).joinpath(f), Path(root).joinpath('save_weights').joinpath(f))
