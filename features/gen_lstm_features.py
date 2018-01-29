#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
时间序列特征
@author: SunnyMarkLiu
@time  : 18-1-29 下午12:58
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from bi_lstm_model import BiLSTM
from sklearn.model_selection import StratifiedKFold
from utils import data_utils
# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float('test_split_percentage', 0.2, 'Percentage of the training data to use for validation')
tf.flags.DEFINE_float('validate_split_percentage', 0.1, 'Percentage of the training data to use for validation')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 9, 'Dimensionality of word embedding (default: 9)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_float("max_learning_rate", 0.001, "Max learning_rate when start training (default: 0.01)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("train_verbose_every_steps", 1, "Show the training info every steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every_steps", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every_steps", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("max_num_checkpoints_to_keep", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("decay_rate", 0.9, "Learning rate decay rate (default: 0.9)")
tf.flags.DEFINE_integer("decay_steps", 10000, "Perform learning rate decay step (default: 10000)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regulaization rate (default: 10000)")

FLAGS = tf.flags.FLAGS

print('Training Parameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


print('---> load train text dataset')
train_seqs, train_y, test_seqs = data_utils.load_action_sequence_label_for_nn()
print('---> build vocabulary according this text dataset')
document_len = np.array([len(x.split(" ")) for x in train_seqs])
print('document_length, max = {}, mean = {}, min = {}'.format(document_len.max(), document_len.mean(),
                                                              document_len.min()))
max_document_length = 1000
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
# Maps documents to sequences of word ids in vocabulary
X_train = np.array(list(vocab_processor.fit_transform(train_seqs)))
# map text to vocabulary index
x_text = np.array(list(vocab_processor.transform(test_seqs)))

vocabulary_size = len(vocab_processor.vocabulary_)
print('built vocabulary size: {:d}'.format(vocabulary_size))

print('---> Split train/validate/test set')
roof_flod = 5
kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=42)

pred_train_full = np.zeros(X_train.shape[0])
pred_test_full = 0
cv_scores = []

test_data_wrapper = data_utils.DataWrapper(x_text, istrain=False, is_shuffle=False)

print('---> build model')
for i, (dev_index, val_index) in enumerate(kf.split(X_train, train_y)):
    print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                            len(val_index)))

    train_y_onehot = [[0, 1] if order_type else [1, 0] for order_type in train_y]
    train_y_onehot = np.array(train_y_onehot)

    x_train, x_valid = X_train[dev_index], X_train[val_index]
    y_train, y_valid = train_y_onehot[dev_index], train_y_onehot[val_index]

    train_data_wrapper = data_utils.DataWrapper(x_train, y_train, istrain=True, is_shuffle=True)
    valid_data_wrapper = data_utils.DataWrapper(x_valid, y_valid, istrain=True, is_shuffle=False)

    with tf.Graph().as_default(), tf.device('/gpu:2'):
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session = tf.Session(config=session_conf)
        with session.as_default():
            bilstm = BiLSTM(embedding_dim=5,
                            sequence_length=max_document_length,
                            label_size=2,
                            hidden_size=50,
                            vocabulary_size=vocabulary_size,
                            embedding_trainable=True,
                            l2_reg_lambda=FLAGS.l2_reg_lambda)
            # Define global training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=bilstm.learning_rate)
            # train_op = optimizer.minimize(bilstm.loss, global_step=global_step)
            grads_and_vars = optimizer.compute_gradients(
                bilstm.loss)  # Compute gradients of `loss` for the variables in `var_list`.
            # some other operation to grads_and_vars, eg. cap the gradient
            # ...
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # Apply gradients to variables.

            # Keep track of gradient values and sparsity distribution (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(name="{}/grad/hist".format(v.name), values=g)
                    # The fraction of zeros in gradient
                    grad_sparsity_summary = tf.summary.scalar("{}/grad/parsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(grad_sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = 'fold_{}'.format(i)
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))

            # Summaries for loss and accuracy
            learning_rate_summary = tf.summary.scalar("learning_rate", bilstm.learning_rate)
            loss_summary = tf.summary.scalar("loss", bilstm.loss)
            accuracy_summary = tf.summary.scalar("accuracy", bilstm.accuracy)

            # Merge Summaries for train
            train_summary_op = tf.summary.merge(
                [learning_rate_summary, loss_summary, accuracy_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(logdir=train_summary_dir, graph=session.graph)

            # Merge Summaries for valid
            valid_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
            valid_summary_writer = tf.summary.FileWriter(logdir=valid_summary_dir, graph=session.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_num_checkpoints_to_keep)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocabulary"))

            # Initialize all variables
            session.run(tf.global_variables_initializer())

            print('---> start training...')
            def train_step(x_batch, y_batch, learning_rate_):
                """ training step """
                feed_dict = {
                    bilstm.sentence: x_batch,
                    bilstm.labels: y_batch,
                    bilstm.learning_rate: learning_rate_
                }
                _, step, summaries, loss_, accuracy_ = session.run([train_op, global_step, train_summary_op,
                                                                    bilstm.loss, bilstm.accuracy],
                                                                   feed_dict=feed_dict)
                if step % FLAGS.train_verbose_every_steps == 0:
                    time_str_ = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print("train {}: step {}, loss {:g}, acc {:g}, learning_rate {:g}"
                          .format(time_str_, step, loss_, accuracy_, learning_rate_))

                train_summary_writer.add_summary(summaries, step)


            def valid_step(x_batch, y_batch, valid_writer=None):
                """ validate step """
                feed_dict = {
                    bilstm.sentence: x_batch,
                    bilstm.labels: y_batch
                }
                summaries, loss_, accuracy_ = session.run([valid_summary_op,
                                                           bilstm.loss, bilstm.accuracy],
                                                          feed_dict=feed_dict)
                if valid_writer:
                    valid_writer.add_summary(summaries, current_step)
                return loss_, accuracy_


            max_learning_rate = FLAGS.max_learning_rate
            decay_rate = FLAGS.decay_rate
            decay_steps = FLAGS.decay_steps

            total_train_steps = FLAGS.epochs * (x_train.shape[0] // FLAGS.batch_size)
            # decay_speed = FLAGS.decay_coefficient * len(y_train) / FLAGS.batch_size

            print('---> total train steps: {}'.format(total_train_steps))
            counter = 0
            start_train_time = time.time()
            for epoch in range(FLAGS.epochs):
                # print('-----> train epoch: {:d}/{:d}'.format(epoch + 1, FLAGS.epochs))
                for i in range(x_train.shape[0] // FLAGS.batch_size):
                    # learning_rate = max_learning_rate * math.pow(decay_rate, int(counter / decay_steps))
                    # counter += 1
                    learning_rate = max_learning_rate

                    batch_x, batch_y = train_data_wrapper.next_batch(FLAGS.batch_size)
                    if len(batch_x) == 0:
                        continue

                    train_step(batch_x, batch_y, learning_rate)
                    current_step = tf.train.global_step(session, global_step)

                    if current_step % FLAGS.evaluate_every_steps == 0:
                        print('---> perform validate')

                        val_x, val_y = valid_data_wrapper.load_all_data()
                        loss, accuracy = valid_step(val_x, val_y, valid_summary_writer)

                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                        print("---> valid {}: step: {:d}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy))

                    if current_step % FLAGS.checkpoint_every_steps == 0:
                        path = saver.save(session, checkpoint_prefix, global_step=global_step)

            end_train_time = time.time()
            print('---> predict valid and test')
            pred_valid = session.run(bilstm.prediction_probs, feed_dict={bilstm.sentence: valid_data_wrapper.load_all_data()[0]})
            pred_test = session.run(bilstm.prediction_probs, feed_dict={bilstm.sentence: test_data_wrapper.load_all_data()[0]})

            pred_train_full[val_index] = pred_valid[:, 1]
            pred_test_full += pred_test[:, 1]
