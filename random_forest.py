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

