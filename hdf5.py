from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf
import h5py  # pylint: disable=g-bad-import-order


X_FEATURE = 'x'  # Name of the input feature.


def main(unused_argv):
  # Load dataset.
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

if __name__ == '__main__':
  tf.app.run()