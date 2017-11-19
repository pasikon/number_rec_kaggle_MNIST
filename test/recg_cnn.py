import os
import sys
import inspect

import tensorflow as tf

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from test.imgload.imgload import random_mini_batches
from test.imgload.imgload import load_train_and_dev_mnist

import matplotlib.pyplot as plt
import matplotlib.cm as cm

tf.logging.set_verbosity(tf.logging.INFO)


def display(img):
    # (784) => (28,28)
    one_image = img.reshape(28, 28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()


def cnn_model_fn(features, keep_probability, mode):
    x_image = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv1_pool = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=conv1_pool,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv2_pool = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # dense layer
    pool2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # keep_prob = tf.placeholder(tf.float32)
    dropout = tf.layers.dropout(inputs=dense, rate=keep_probability, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits


X_train, Y_train, X_test, Y_test = load_train_and_dev_mnist()

x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
training_mode = tf.placeholder(tf.bool, name="training_mode_bool")

y_conv = cnn_model_fn(x, keep_prob, training_mode)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
starter_learning_rate = 1e-4

with tf.name_scope("learning_rate"):
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_tensor, 150000, 0.96)
    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(cross_entropy, global_step=global_step_tensor)

# valMon = ValidationMonitor(x=X_train.T, y=Y_train.T, every_n_steps=300)
# logHook = LoggingTensorHook(every_n_iter=300, tensors={learning_rate})

# tf.estimator.EstimatorSpec(train_op=train_step, loss=cross_entropy, training_hooks={logHook})

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(max_to_keep=20)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_with_summaries_cnn/train', sess.graph)
    test_writer_dev = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_with_summaries_cnn/test_dev')
    test_writer_train = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_with_summaries_cnn/test_train')

    for epoch in range(500):
        epoch_cost = 0.
        m = X_train.shape[1]
        minibatch_size = 128
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

        for minibatch in minibatches:
            xt = minibatch[0].T
            yt = minibatch[1].T
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(train_step, feed_dict={x: xt, y_: yt, keep_prob: 0.40, training_mode: True},
                     run_metadata=run_metadata, options=run_options)
            train_writer.add_run_metadata(run_metadata, tf.train.global_step(sess, global_step_tensor))

            minibatch_cost = \
                sess.run(cross_entropy, feed_dict={x: xt, y_: yt, keep_prob: 1, training_mode: False})

            epoch_cost += minibatch_cost / num_minibatches

        if epoch % 5 == 0:
            print("--------------------------------------------------------------------------------------------------")
            print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))
            summary, dev_accuracy = \
                sess.run([merged, accuracy], feed_dict={x: X_test.T, y_: Y_test.T, keep_prob: 1.0, training_mode: False})
            test_writer_dev.add_summary(summary, tf.train.global_step(sess, global_step_tensor))
            print("EPOCH {}, DEV accuracy {}".format(epoch, dev_accuracy))
            summary, train_accuracy = \
                sess.run([merged, accuracy], feed_dict={x: X_train.T, y_: Y_train.T, keep_prob: 1.0, training_mode: False})
            test_writer_train.add_summary(summary, tf.train.global_step(sess, global_step_tensor))
            print("EPOCH {}, TRAIN accuracy {}".format(epoch, train_accuracy))
            print("EPOCH COST: " + str(epoch_cost))

        if epoch % 10 == 0:
            saver.save(sess=sess, save_path=os.path.join("modelz/", "model_chkp"), global_step=global_step_tensor)
