#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Codes from https://github.com/bamos/dcgan-completion.tensorflow/blob/master/simple-distributions.py

create some simple distributions plots
"""
# !/usr/bin/env python3

import numpy as np
from scipy.stats import norm

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')
import matplotlib.mlab as mlab

np.random.seed(0)
### 绘制一个正态分布的概率密度函数图###
# 生成数据 X范围是(-3,3),步进为0.001, Y的范围是(0,1)
X = np.arange(-3, 3, 0.001)
Y = norm.pdf(X, 0, 1)
# 绘制
fig = plt.figure()
plt.plot(X, Y)
plt.tight_layout()
plt.savefig("./images/normal-pdf.png")

### 绘制从正态分布采样的 1D 散点图例子 ###
nSamples = 35
# np.random.normal 是从正态分布中随机采样指定数量的样本,这里指定 35个
X = np.random.normal(0, 1, nSamples)
Y = np.zeros(nSamples)
fig = plt.figure(figsize=(7, 3))
# 绘制散点图
plt.scatter(X, Y, color='k')
plt.xlim((-3, 3))
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
plt.savefig("./images/normal-samples.png")

### 绘制从正态分布采样的 2D 散点图例子###

delta = 0.025
# 设置 X,Y 的数值范围和步长值，分别生成 240个数
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
print('x shape', x.shape)
# 根据坐标向量来生成坐标矩阵
X, Y = np.meshgrid(x, y)  # X, Y shape: (240, 240)

print('X shape', X.shape)
print('Y shape', Y.shape)
# Bivariate Gaussian distribution for equal shape *X*, *Y*
# 等形状的双变量高斯分布
Z = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)  # Z shape (240, 240)
print('Z shape', Z.shape)

plt.figure()
# 绘制环形图轮廓
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)

nSamples = 200
mean = [0, 0]
cov = [[1, 0], [0, 1]]
# 从多元正态分布中采样，得到结果图中的黑点例子
X, Y = np.random.multivariate_normal(mean, cov, nSamples).T
plt.scatter(X, Y, color='k')

plt.savefig("./images/normal-2d.png")
