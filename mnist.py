print ("start")

import tensorflow as tf

from keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
# 归一化，0-255不太方便神经网络进行计算，因此将范围缩小到0—1
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255


