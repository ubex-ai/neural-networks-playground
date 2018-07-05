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

  # Note that we are saving and load iris data as h5 format as a simple
  # demonstration here.
  h5f = h5py.File('/tmp/test_hdf5.h5', 'w')
  h5f.create_dataset('X_train', data=x_train)
  h5f.create_dataset('X_test', data=x_test)
  h5f.create_dataset('y_train', data=y_train)
  h5f.create_dataset('y_test', data=y_test)
  h5f.close()

if __name__ == '__main__':
  tf.app.run()