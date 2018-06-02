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


def maybe_download_iris_data(file_name, download_url):
  """Downloads the file and returns the number of data."""
  if not os.path.exists(file_name):
    urlretrieve(download_url, file_name)

  # The first line is a comma-separated string. The first one is the number of
  # total data in the file.
  with open(file_name, 'r') as f:
    first_line = f.readline()
  num_elements = first_line.split(',')[0]
  return int(num_elements)

