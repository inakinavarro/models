# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

def _print_confusion_matrices(labels, predictions):
  num_labels = labels.shape[1]
  threshold = 0.5
  predictions[predictions >= threshold] = 1
  predictions[predictions < threshold] = 0

  for i in range(num_labels):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(predictions[:,i] == 1, labels[:,i] == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(predictions[:,i] == 0, labels[:,i] == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(predictions[:,i] == 1, labels[:,i] == 0)) 
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(predictions[:,i] == 0, labels[:,i] == 1))
    total_neg = TN + FP
    total_pos = FN + TP
    print('Label', i)
    print('------')
    print(TN, FP)
    print(FN, TP)
    print('------')
    print('%.2f %.2f'% (TN/total_neg, FP/total_neg))
    print('%.2f %.2f'% (FN/total_pos, TP/total_pos))
    print('')

def _eval_once(saver, summary_writer, sigmoid_predictions, summary_op,
            num_examples_dataset, labels_op, num_classes):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    preditcions: sigmoid predictions.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Successfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      if FLAGS.num_examples != -1:
        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      else:
        num_iter = int(math.ceil(num_examples_dataset / FLAGS.batch_size))
        
      # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      predictions_all = np.empty((0, num_classes))
      labels_all = np.empty((0, num_classes))
      while step < num_iter and not coord.should_stop():
        predictions, labels = sess.run([sigmoid_predictions, labels_op])
        predictions_all = np.append(predictions_all, predictions, axis=0)
        labels_all = np.append(labels_all, labels, axis=0)

        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute precision @ 1.
      # precision_at_1 = count_top_1 / total_sample_count
      # recall_at_5 = count_top_5 / total_sample_count
      # print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
      #       (datetime.now(), precision_at_1, recall_at_5, total_sample_count))
      _print_confusion_matrices(labels_all, predictions_all)
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      # summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      # summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels = image_processing.inputs(dataset)

    # Number of classes in the Dataset label.
    num_classes = dataset.num_classes()

    # Number of examples in the Dataset.
    num_examples_dataset = dataset.num_examples_per_epoch()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes)

    # Calculate predictions.
    sigmoid_predictions = tf.nn.sigmoid(logits)
    labels_op = tf.identity(labels)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      _eval_once(saver, summary_writer, sigmoid_predictions, summary_op,
                  num_examples_dataset, labels_op, num_classes)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
