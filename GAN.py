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
from datetime import datetime
from datetime import timedelta

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

            # get number of batches for a single epoch 计算每个 epoch 总共需要训练的迭代次数，也是 batch 数量,此处为70000整除64=1093
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
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

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

        # get loss for discriminator 计算判别器的损失函数,采用的是交叉熵函数
        # d_loss_real=log(sigmoid(D_real_logits)) 等价于 d_loss_real= log(D(x))
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        # d_loss_fake = log(sigmoid(D_fake_logits)) 等价于 d_loss_fake=log(D(G(z)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator 生成器的损失函数
        # g_loss=-log(sigmoid(D_fake_logits))等价于g_loss=-log(D(G(z))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        # 按照判别器和生成器将训练变量分为两组
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers 优化器用于计算梯度，进行反向传播，更新参数，采用的是 Adam 优化器
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(
                self.d_loss, var_list=d_vars)
            # 生成器采用的学习率是判别器的 5 倍
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1).minimize(
                self.g_loss, var_list=g_vars)

        """" Testing """
        # for test 用于测试, 所有设置 is_training=False, reuse=True
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        # 记录训练中的一些参数，用于在 Tensorboard 中进行观察和可视化
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations 利用 merge 函数将 summary 的运算都集合在一起
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    # 训练模型
    def train(self):

        # initialize all variables 初始化各个变量
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        # 创建随机噪声 z，且是从均匀分布为[-1,1]中采样得到，维度是 [64, 62]
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model 保存训练模型的参数和网络结构
        self.saver = tf.train.Saver()

        # summary writer 训练记录会保存在 log_dir 文件夹
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits 判断是否需要载入之前训练的模型，恢复之前的训练
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network 将数据输入 D 网络，更新梯度，并更新 tensorboard 观察和可视化的内容，注意需要输入图片和噪声
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network 将数据输入 G 网络，更新梯度，并更新 tensorboard 观察和可视化的内容，注意需要输入噪声
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status 输出训练的状态，包括当前训练迭代次数、训练时间、G 和 D 网络的 loss
                counter += 1
                # 设置每 100 次输出一次
                if np.mod(counter, 100) == 0:
                    # 计算当前训练耗时
                    elapsed = time.time() - start_time
                    elapsed = str(timedelta(seconds=elapsed))
                    print("%s Eplased: [%s], Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f"
                          % (datetime.now(), elapsed, epoch, idx, self.num_batches, d_loss, g_loss))

                # save training results for every 300 steps 设置每 300 次迭代保存训练结果
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    # 此处计算生成图片的小框图片的排布，本处为8×8排布
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, idx))
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            # 每训练一次 epoch 后，需要重置 start_batch_id 为 0，start_batch_id 只有在加载预训练模型的时候才是非 0 值
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step 保存最后一次训练的模型参数
        self.save(self.checkpoint_dir, counter)

    def generate_images(self, model):
        """generate_images
        Method to generate samples using a pre-trained model
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model))
        # todo

    def generate_tsne(self):
        """generate_tsne
        Method to visualize TSNE with random samples from the ground truth and
        generated distribution. This might help in catching mode collapse. If
        there is an obvious case of mode collapse, then we should see several
        points from the ground truth without any generated samples nearby.
        Purely a sanity check.

        """
        from sklearn.manifold import TSNE
        num_points = 1000
        # todo
        pass

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
