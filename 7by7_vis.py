import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data_No = 4
CE_No = 0
train_7by7CE = np.load("./7by7_CE/train/train_7by7CE.npy")
train_CE = np.load("./7by7_CE/train/train_CE.npy")
train_trueCE = np.load("./7by7_CE/train/train_attr_cms.npy")
trueCE = train_trueCE[data_No, :].ravel()
CE = train_CE[data_No, :].ravel()
attributes_name = pd.read_csv('./data/AWA2/predicates.txt', header=None, sep='\t')

x = np.arange(7)
y = np.arange(7)
X, Y = np.meshgrid(x, y)
X = X.ravel()
Y = Y.ravel()

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111, projection="3d")


ax1.bar3d(X, Y, np.zeros_like(X), 1, 1, train_7by7CE[data_No, :, :, CE_No].ravel())
ax1.set_zlim((train_7by7CE[data_No, :, :, CE_No].min(), train_7by7CE[data_No, :, :, CE_No].max()))
print("CE = ", CE[CE_No])
print("ground_truth_CE = ", trueCE[CE_No])

plt.show()
