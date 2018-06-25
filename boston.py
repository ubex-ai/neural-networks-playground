from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf


def main(unused_argv):
  # Load dataset
  boston = datasets.load_boston()
  x, y = boston.data, boston.target


if __name__ == '__main__':
  tf.app.run()