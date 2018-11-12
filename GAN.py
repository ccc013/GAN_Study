#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

GAN model
Code from https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
"""
from __future__ import division, print_function
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


class GAN(object):
    # 初始化参数
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        # name for checkpoint 保存模型的名字
        self.model_name = "GAN"
        # mnist 和 fashion-mnist 都是图片大小是 28*28 的灰度图片组成的数据集
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector 噪声向量的维度
            self.c_dim = 1

            # train 训练的参数
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test 测试参数
            self.sample_num = 64  # number of generated images to be saved

            # load mnist 加载数据集
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch 计算每个 epoch 总共需要训练的迭代次数，也是 batch 数量
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    # 鉴别器
    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        '''

        :param x: 输入 x [64, 28, 28, 1]
        :param is_training:
        :param reuse:
        :return:
        '''
        with tf.variable_scope("discriminator", reuse=reuse):
            # 经过这步卷积后，[64, 28, 28, 1]->[64, 14, 14, 64]
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            # 经过这步卷积后，[64, 14, 14, 64]->[64, 7, 7, 128]
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            # 调整维度后，net 维度是[64, 6272]
            net = tf.reshape(net, [self.batch_size, -1])
            # 全连接操作，[64, 6272] -> [64, 1024]
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            # 全连接操作，[64, 1024]->[64, 1]，这一步得到输出结果
            out_logit = linear(net, 1, scope='d_fc4')
            # 采用 sigmoid 激活函数，将输出范围限制在 [0,1]
            out = tf.nn.sigmoid(out_logit)
            # 返回三个输出
            return out, out_logit, net

    # 生成器
    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        '''

        :param z: 输入噪声 z [64, 62]
        :param is_training:
        :param reuse:
        :return:
        '''
        with tf.variable_scope("generator", reuse=reuse):
            # 全连接操作，[64, 62]->[64, 1024]
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            # 全连接操作, [64, 1024]->[64, 6272] 6272=128 * 7 * 7
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            # 调整维度，[64, 6272]->[64, 7, 7, 128]
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            # 反卷积，[64, 7, 7, 128]->[64, 14, 14, 64]
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))
            # 反卷积 [64, 14, 14, 64]->[64, 28, 28, 1]，再采用 sigmoid 激活函数，限制输出在 [0,1]
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out

    # 建立 GAN 模型
    def build_model(self):
        # some parameters 建立模型需要设置的参数
        # 对于mnist数据集，图片大小为[28,28,1]，此处用list列表存储
        image_dims = [self.input_height, self.input_width, self.c_dim]
        # batch 大小，默认是 64
        bs = self.batch_size

        """ Graph Input 图输入设置"""
        # images 设置 placeholder 用于图片输入，shape=[64,28,28,1]
        self.inputs = tf.placeholder(tf.float32, [bs]+image_dims, name='real_images')

        # noises 噪声 [64, 62]
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        '''Loss Function 损失函数'''
        # output of D for real images 给定真实图片输入的判别器的输出
        # D_real [64, 1],且数值是在[0,1]范围，D_real_logits [64, 1024], 没有采用sigmoid激活函数
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images 给定假样本输入的判别器的输出
        # 获取生成器的生成图片结果, G [64, 28, 28, 1]
        G = self.generator(self.z, is_training=True, reuse=False)
        # D_fake [64, 1],且数值是在[0,1]范围，D_fake_logits [64, 1024], 没有采用sigmoid激活函数
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator 计算判别器的损失函数
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real))
        )