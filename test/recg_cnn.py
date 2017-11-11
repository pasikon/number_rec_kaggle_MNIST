import math
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from test.imgload.imgload import loaddta

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from test.convolutional.convlayers import bias_variable
from test.convolutional.convlayers import conv2d
from test.convolutional.convlayers import conv_layer_batch_norm
from test.convolutional.convlayers import full_layer
from test.convolutional.convlayers import max_pool_2x2


def display(img):
    # (784) => (28,28)
    one_image = img.reshape(28, 28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()


X_loaded, Y_loaded, _ = loaddta('input/train.csv')

X_train = X_loaded[:, 0:39000]
Y_train = Y_loaded[:, 0:39000]
X_test = X_loaded[:, 39001:41999]
Y_test = Y_loaded[:, 39001:41999]

print('X_train shape:' + str(X_train.shape))
print('Y_train shape:' + str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
print('Y_test shape:' + str(Y_test.shape))

# display(X_train[:, 3905])

x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
training_mode = tf.placeholder(tf.bool)

x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer_batch_norm(x_image, shape=[5, 5, 1, 32], training_mode=training_mode)
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer_batch_norm(conv1_pool, shape=[5, 5, 32, 64], training_mode=training_mode)
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
starter_learning_rate = 1e-4
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_tensor, 150000, 0.96)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step_tensor)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("conv1: " + str(conv1))
print("conv1_pool: " + str(conv1_pool))
print("conv2: " + str(conv2))
print("conv2_pool: " + str(conv2_pool))
print("conv2_flat: " + str(conv2_flat))
print("full_1: " + str(full_1))
print("y_conv: " + str(y_conv))


def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Produces shuffled list of minibatches
    :param X:
    :param Y:
    :param mini_batch_size:
    :return: List of tuples (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling end case where last mini-batch < mini_batch_size
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


saver = tf.train.Saver(max_to_keep=20)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(500):
        epoch_cost = 0.
        m = X_train.shape[1]
        minibatch_size = 128
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

        for minibatch in minibatches:
            xt = minibatch[0].T
            yt = minibatch[1].T
            xt = xt.astype(np.float32)

            sess.run(train_step, feed_dict={x: xt, y_: yt, keep_prob: 0.40, training_mode: True})

            minibatch_cost = \
                sess.run(cross_entropy, feed_dict={x: xt, y_: yt, keep_prob: 1, training_mode: False})

            epoch_cost += minibatch_cost / num_minibatches

        if epoch % 5 == 0:
            print("--------------------------------------------------------------------------------------------------")
            print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))
            dev_accuracy = \
                sess.run(accuracy, feed_dict={x: X_test.T, y_: Y_test.T, keep_prob: 1.0, training_mode: False})
            print("EPOCH {}, DEV accuracy {}".format(epoch, dev_accuracy))
            train_accuracy = \
                sess.run(accuracy, feed_dict={x: X_train.T, y_: Y_train.T, keep_prob: 1.0, training_mode: False})
            print("EPOCH {}, TRAIN accuracy {}".format(epoch, train_accuracy))
            print("EPOCH COST: " + str(epoch_cost))

        if epoch % 5 == 0:
            saver.save(sess=sess, save_path=os.path.join("modelz/", "model_chkp"), global_step=global_step_tensor)
