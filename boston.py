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

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1:])]
  regressor = tf.estimator.DNNRegressor(
      feature_columns=feature_columns, hidden_units=[10, 10])

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True)
  regressor.train(input_fn=train_input_fn, steps=2000)


if __name__ == '__main__':
  tf.app.run()