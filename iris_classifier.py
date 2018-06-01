from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves.urllib.request import urlretrieve

import tensorflow as tf

# Data sets
IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

