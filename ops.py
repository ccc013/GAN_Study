#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

Most codes from https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/ops.py
"""
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

if "concat_v2" in dir(tf):
    # TF 1.0 版本前采用的函数
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    # TF 1.0+ 版本后的函数
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


# BatchNorm 层
def bn(x, is_training, scope):
    return tf.layers.batch_normalization(x,
                                         axis=-1,
                                         momentum=0.99,
                                         epsilon=1e-3,
                                         center=True,
                                         scale=True,
                                         training=is_training,
                                         trainable=True,
                                         name=scope,
                                         reuse=None
                                         )


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def conv_cond_concat(x, y):
    """
    Concatenate conditioning vector on feature map axis.
    将条件向量 y 和 x 在 feature map 的维度上连接
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


# 卷积层
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    '''

    :param input_: 输入数据
    :param output_dim: 输出 feature map 的 数量
    :param k_h: 卷积核的高
    :param k_w: 卷积核的宽
    :param d_h: 步长strides的高
    :param d_w: 步长strides的宽
    :param stddev: 初始化卷积核所需的方差
    :param name: 卷积层名字
    :return conv: 返回经过卷积操作后的输出 feature maps
    '''
    with tf.variable_scope(name):
        input_dims = input_.get_shape()[-1]
        w = tf.get_variable('weights', [k_h, k_w, input_dims, output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# 反卷积层
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, with_w=False, name='deconv2d'):
    '''

    :param input_: 输入数据
    :param output_shape: 输出 feature map 的 数量
    :param k_h: 卷积核的高
    :param k_w: 卷积核的宽
    :param d_h: 步长strides的高
    :param d_w: 步长strides的宽
    :param stddev: 初始化卷积核所需的方差
    :param with_w: 是否返回权重 weights
    :param name: 反卷积层名字
    :return:
    '''
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


# leaky relu 激活层
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.nn.leaky_relu(x, leak, name=name)


# 全连接层
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("biases", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
