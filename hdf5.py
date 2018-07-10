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

  h5f = h5py.File('/tmp/test_hdf5.h5', 'r')
  x_train = np.array(h5f['X_train'])
  x_test = np.array(h5f['X_test'])
  y_train = np.array(h5f['y_train'])
  y_test = np.array(h5f['y_test'])

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(x_train).shape[1:])]
  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=200)

  # Predict.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class_ids'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()