import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.framework import ops

from test.imgload.imgload import loaddta

from test.convolutional.convlayers import bias_variable
from test.convolutional.convlayers import conv2d
from test.convolutional.convlayers import conv_layer
from test.convolutional.convlayers import full_layer
from test.convolutional.convlayers import max_pool_2x2


X_loaded, Y_loaded = loaddta('../input/train.csv')

X_train = X_loaded[:, 0:39000]
Y_train = Y_loaded[:, 0:39000]
X_test = X_loaded[:, 39001:41999]
Y_test = Y_loaded[:, 39001:41999]

print('X_train shape:' + str(X_train.shape))
print('Y_train shape:' + str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
print('Y_test shape:' + str(Y_test.shape))


x_image = tf.reshape(x, [-1, 28, 28, 1])


