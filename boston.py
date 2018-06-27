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

  # Split dataset into train / test
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=42)

  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train = scaler.fit_transform(x_train)

if __name__ == '__main__':
  tf.app.run()