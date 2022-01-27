import os
import configure
from pathlib import Path


class ClassNameIndexMap:
    def __init__(self):
        f = open(Path(configure.dataset_dir).joinpath(configure.dataset).joinpath('class_order.txt'), 'r')
        self.name = {}
        self.index = {}
        line = f.readline()
        while line != '':
            self.name[line.split()[0]] = int(line.split()[1])
            self.index[int(line.split()[1])] = line.split()[0]
            line = f.readline()

    def index(self, name):
        return self.name[name]

    def name(self, index):
        return self.index[index]

    def name_list(self):
        name_list = []
        for i in range(configure.class_num):
            name_list.append(self.index[i])
        return name_list
