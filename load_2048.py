import os
import numpy as np
from pathlib import Path


class LoadFeature:
    def __init__(self, path):
        self.feature_dir = Path(path)
        walk_generator = os.walk(self.feature_dir)
        _, directory, _ = next(walk_generator)
        directory.sort()
        self.features = {}
        for d in directory:
            walk_generator2 = os.walk(self.feature_dir.joinpath(d))
            root, _, files = next(walk_generator2)
            files.sort()
            for f in files:
                if Path(f).suffix == '.npy':
                    self.features[int(Path(f).stem)] = Path(root).joinpath(f)
        self.size = len(self.features)

    def __getitem__(self, index):
        if index < self.size:
            return self.features[index]


if __name__ == '__main__':
    lf = LoadFeature('E:/Model/AWA2/tfrecord/none/random_crop/data')
    features = np.zeros((lf.size, 2048))
    for i in range(lf.size):
        features[i, :] = np.load(str(lf[i]))
    stop = 1
