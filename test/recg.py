import math
import numpy as np
import pandas as pd

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.python.framework import ops

plt.interactive(False)


# def display(img):
#     # (784) => (28,28)
#     one_image = img.reshape(image_width, image_height)
#     plt.axis('off')
#     plt.imshow(one_image, cmap=cm.binary)
#     plt.show()


def one_hot_matrix(labels, C):
    c = tf.constant(C, name="C")
    oht = tf.one_hot(indices=labels, depth=c, axis=1)
    with tf.Session() as sess:
        yoh = sess.run(oht)
    yoh = yoh[0]
    return yoh


def cost(logits, labels):
    z = tf.placeholder(tf.float32, name="logitsT")
    y = tf.placeholder(tf.float32, name="labelsT")
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    sess = tf.Session()
    cost = sess.run(cost, feed_dict={z: logits, y: labels})
    sess.close()
    return cost


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")
    keepprob_A1 = tf.placeholder(tf.float32)
    keepprob_A2 = tf.placeholder(tf.float32)
    return X, Y, keepprob_A1, keepprob_A2


def initialize_parameters():
    W1 = tf.get_variable("W1", [25, 784], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [17, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [17, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [12, 17], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [12, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [10, 12], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [10, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}

    return parameters


def forward_propagation(X, parameters, keepprob_A1, keepprob_A2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    relu1_dropout = tf.nn.dropout(A1, keep_prob=keepprob_A1)

    Z2 = tf.matmul(W2, relu1_dropout) + b2
    A2 = tf.nn.relu(Z2)
    relu2_dropout = tf.nn.dropout(A2, keep_prob=keepprob_A2)

    Z3 = tf.matmul(W3, relu2_dropout) + b3
    A3 = tf.nn.relu(Z3)

    Z4 = tf.matmul(W4, A3) + b4

    return Z4


def compute_cost(Z4, Y):
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def random_mini_batches(X, Y, mini_batch_size=64):

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


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.00009,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y, keepprob_A1, keepprob_A2 = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z4 = forward_propagation(X, parameters, keepprob_A1, keepprob_A2)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z4, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch. Run the session to execute the "optimizer"
                # and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keepprob_A1: 0.8, keepprob_A2: 0.7})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keepprob_A1: 1, keepprob_A2: 1}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keepprob_A1: 1, keepprob_A2: 1}))

        return parameters


def loaddta(csvp):
    data = pd.read_csv(csvp)
    # print('data({0[0]},{0[1]})'.format(data.shape))
    # print(data.head())
    images = data.iloc[:, 1:].values
    images = images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    images = np.multiply(images, 1.0 / 255.0)
    image_size = images.shape[1]
    print('image_size => {0}'.format(image_size))

    # in this case all images are square
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

    print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

    # display(images[655])

    X = images.T
    # print("X shape:" + str(X.shape))
    y = data.iloc[:, 0:1].values.T
    # print("y shape:" + str(y.shape))
    YOH = one_hot_matrix(y, 10)
    # print("YOH shape:" + str(YOH.shape))
    return X, YOH


# read training data from CSV file
# data = pd.read_csv('../input/train.csv')
# test_data = pd.read_csv('../input/train.csv')

X_loaded, Y_loaded = loaddta('../input/train.csv')
# X_test, Y_test = loaddta('../input/test.csv')

X_train = X_loaded[:, 0:39000]
Y_train = Y_loaded[:, 0:39000]
X_test = X_loaded[:, 39001:41999]
Y_test = Y_loaded[:, 39001:41999]

print('X_train shape:' + str(X_train.shape))
print('Y_train shape:' + str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
print('Y_test shape:' + str(Y_test.shape))

parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=210)
