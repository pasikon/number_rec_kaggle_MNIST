import pandas as pd
import tensorflow as tf
import numpy as np


def loaddta(csvp):
    data = pd.read_csv(csvp)
    # print('data({0[0]},{0[1]})'.format(data.shape))
    # print(data.head())
    images = data.iloc[:, 1:].values
    images = images.astype(np.float32)
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
    return X, YOH, y


def load_train_and_dev_mnist():
    X_loaded, Y_loaded, _ = loaddta('input/train.csv')

    X_train = X_loaded[:, 0:39000]
    Y_train = Y_loaded[:, 0:39000]
    X_test = X_loaded[:, 39001:41999]
    Y_test = Y_loaded[:, 39001:41999]

    print('X_train shape:' + str(X_train.shape))
    print('Y_train shape:' + str(Y_train.shape))
    print('X_test shape:' + str(X_test.shape))
    print('Y_test shape:' + str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


def one_hot_matrix(labels, C):
    c = tf.constant(C, name="C")
    oht = tf.one_hot(indices=labels, depth=c, axis=1)
    with tf.Session() as sess:
        yoh = sess.run(oht)
    yoh = yoh[0]
    return yoh