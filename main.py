#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Code from https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/main.py
"""
import os
from datetime import datetime

# GAN Variants
from GAN import GAN

from utils import *
from options import *
import tensorflow as tf

"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # open session
    with tf.Session(config=config) as sess:
        # declare instance for GAN
        if args.gan_type == 'GAN':
            gan = GAN(sess,
                      epoch=args.epoch,
                      batch_size=args.batch_size,
                      z_dim=args.z_dim,
                      dataset_name=args.dataset,
                      checkpoint_dir=args.checkpoint_dir,
                      result_dir=args.result_dir,
                      log_dir=args.log_dir)
        else:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print("{} Training finished!".format(datetime.now()))

        # visualize learned generator
        gan.visualize_results(args.epoch - 1)
        print("{} Testing finished!".format(datetime.now()))


if __name__ == '__main__':
    main()
