from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy

from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.platform import app

FLAGS = None


def build_estimator(model_dir):
  """Build an estimator."""
  params = tensor_forest.ForestHParams(
      num_classes=10,
      num_features=784,
      num_trees=FLAGS.num_trees,
      max_nodes=FLAGS.max_nodes)
  graph_builder_class = tensor_forest.RandomForestGraphs
  if FLAGS.use_training_loss:
    graph_builder_class = tensor_forest.TrainingLossForest
  return random_forest.TensorForestEstimator(
      params, graph_builder_class=graph_builder_class, model_dir=model_dir)


def train_and_eval():
  """Train and evaluate the model."""
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print('model directory = %s' % model_dir)

  est = build_estimator(model_dir)

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

  train_input_fn = numpy_io.numpy_input_fn(
      x={'images': mnist.train.images},
      y=mnist.train.labels.astype(numpy.int32),
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)
  est.fit(input_fn=train_input_fn, steps=None)

  metric_name = 'accuracy'
  metric = {
      metric_name:
          metric_spec.MetricSpec(
              eval_metrics.get_metric(metric_name),
              prediction_key=eval_metrics.get_prediction_key(metric_name))
  }

  test_input_fn = numpy_io.numpy_input_fn(
      x={'images': mnist.test.images},
      y=mnist.test.labels.astype(numpy.int32),
      num_epochs=1,
      batch_size=FLAGS.batch_size,
      shuffle=False)

  results = est.evaluate(input_fn=test_input_fn, metrics=metric)
  for key in sorted(results):
    print('%s: %s' % (key, results[key]))


def main(_):
  train_and_eval()

