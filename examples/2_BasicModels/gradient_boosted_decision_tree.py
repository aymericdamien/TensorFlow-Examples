""" GBDT (Gradient Boosted Decision Tree).



Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.learn import learn_runner

# Ignore all GPUs, tf random forest does not benefit from it.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False,
                                  source_url='http://yann.lecun.com/exdb/mnist/')

# Parameters
log_dir = "/tmp/tf_gbdt"
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

# GBDT Parameters
learning_rate = 0.1
l1_regul = 0.
l2_regul = 1.
examples_per_layer = 1000
num_trees = 10
max_depth = 4


def get_input_fn(x, y):
    """Input function over MNIST data."""

    def input_fn():
        images_batch, labels_batch = tf.train.shuffle_batch(
                tensors=[x, y],
                batch_size=batch_size,
                capacity=batch_size * 10,
                min_after_dequeue=batch_size * 2,
                enqueue_many=True,
                num_threads=4)
        features_map = {"images": images_batch}
        return features_map, labels_batch

    return input_fn


learner_config = learner_pb2.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.num_classes = num_classes
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth

growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER
learner_config.growing_mode = growing_mode
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)

learner_config.multi_class_strategy = (
    learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)

# Create a TF Boosted trees estimator that can take in custom loss.
estimator = GradientBoostedDecisionTreeClassifier(
    learner_config=learner_config,
    n_classes=num_classes,
    examples_per_layer=examples_per_layer,
    model_dir=log_dir,
    num_trees=num_trees,
    center_bias=False,
    config=run_config)


def _make_experiment_fn(output_dir):
  """Creates experiment for gradient boosted decision trees."""
  train_input_fn = get_input_fn(mnist.train.images,
                            mnist.train.labels.astype(np.int32))
  eval_input_fn = get_input_fn(mnist.test.images,
                           mnist.test.labels.astype(np.int32))

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=None,
      eval_steps=1,
      eval_metrics=None)

# Training
learn_runner.run(
      experiment_fn=_make_experiment_fn,
      output_dir=log_dir,
      schedule="train_and_evaluate")


# Accuracy
test_input_fn = get_input_fn(
    mnist.test.images, mnist.test.labels.astype(np.int32))
results = estimator.predict(x=mnist.test.images)

acc = 0.
n = 0
for i, r in enumerate(results):
    if np.argmax(r['probabilities']) == int(mnist.test.labels[i]):
        acc += 1
    n += 1

print(acc / n)
