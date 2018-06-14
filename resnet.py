from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt

import numpy as np
import tensorflow as tf


N_DIGITS = 10  # Number of digits.
X_FEATURE = 'x'  # Name of the input feature.


def res_net_model(features, labels, mode):
  """Builds a residual network."""

  # Configurations for each bottleneck group.
  BottleneckGroup = namedtuple('BottleneckGroup',
                               ['num_blocks', 'num_filters', 'bottleneck_size'])
  groups = [
      BottleneckGroup(3, 128, 32), BottleneckGroup(3, 256, 64),
      BottleneckGroup(3, 512, 128), BottleneckGroup(3, 1024, 256)
  ]

  x = features[X_FEATURE]
  input_shape = x.get_shape().as_list()

  # Reshape the input into the right shape if it's 2D tensor
  if len(input_shape) == 2:
    ndim = int(sqrt(input_shape[1]))
    x = tf.reshape(x, [-1, ndim, ndim, 1])

  training = (mode == tf.estimator.ModeKeys.TRAIN)

  # First convolution expands to 64 channels
  with tf.variable_scope('conv_layer1'):
    net = tf.layers.conv2d(
        x,
        filters=64,
        kernel_size=7,
        activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)

  # Max pool
  net = tf.layers.max_pooling2d(
      net, pool_size=3, strides=2, padding='same')

  # First chain of resnets
  with tf.variable_scope('conv_layer2'):
    net = tf.layers.conv2d(
        net,
        filters=groups[0].num_filters,
        kernel_size=1,
        padding='valid')

  # Create the bottleneck groups, each of which contains `num_blocks`
  # bottleneck groups.
  for group_i, group in enumerate(groups):
    for block_i in range(group.num_blocks):
      name = 'group_%d/block_%d' % (group_i, block_i)

      # 1x1 convolution responsible for reducing dimension
      with tf.variable_scope(name + '/conv_in'):
        conv = tf.layers.conv2d(
            net,
            filters=group.num_filters,
            kernel_size=1,
            padding='valid',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=training)

      with tf.variable_scope(name + '/conv_bottleneck'):
        conv = tf.layers.conv2d(
            conv,
            filters=group.bottleneck_size,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=training)

      # 1x1 convolution responsible for restoring dimension
      with tf.variable_scope(name + '/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = tf.layers.conv2d(
            conv,
            filters=input_dim,
            kernel_size=1,
            padding='valid',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=training)

      # shortcut connections that turn the network into its counterpart
      # residual function (identity shortcut)
      net = conv + net
