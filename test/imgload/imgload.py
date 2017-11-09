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


def one_hot_matrix(labels, C):
    c = tf.constant(C, name="C")
    oht = tf.one_hot(indices=labels, depth=c, axis=1)
    with tf.Session() as sess:
        yoh = sess.run(oht)
    yoh = yoh[0]
    return yoh