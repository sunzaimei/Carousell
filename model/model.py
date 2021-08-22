from abc import ABC

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.python.keras.models import Sequential

from config import class_no


class LeNet(tf.keras.Model, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2Dv1 = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu)
        self.conv2Dv2 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu)
        self.Densev1 = tf.keras.layers.Dense(1024, activation='relu')
        self.Densev2 = tf.keras.layers.Dense(12, activation=tf.nn.softmax)

    def call(self, x, is_training=True, **kwargs):
        # x = Input(shape=(None, None, 3))
        # x = tf.reshape(x, shape=[-1, 64, 64, 3])
        net = self.conv2Dv1(x)
        net = tf.keras.layers.MaxPool2D(2, 2)(net)
        net = self.conv2Dv2(net)
        net = tf.keras.layers.MaxPool2D(2, 2)(net)
        net = tf.keras.layers.Flatten()(net)
        net = self.Densev1(net)
        net = tf.keras.layers.Dropout(rate=0.4)(net, training=is_training)
        net = self.Densev2(net)
        # return net
        return Model(x, net, name="lenet")


def lenet(is_training):
    from config import input_width, input_height
    x = Input(shape=(input_height, input_width, 3))
    # x = tf.reshape(x, shape=[-1, input_height, input_width, 3])
    net = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu)(x)
    net = tf.keras.layers.MaxPool2D(2, 2)(net)
    net = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu)(net)
    net = tf.keras.layers.MaxPool2D(2, 2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(1024, activation='relu')(net)
    net = tf.keras.layers.Dropout(rate=0.4)(net, training=is_training)
    net = tf.keras.layers.Dense(12, activation=tf.nn.softmax)(net)
    return Model(x, net, name="lenet")

def simpleNet():
    from config import input_width, input_height
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(input_width, input_height, 3)))
    model.add(MaxPool2D())

    # model.add(Conv2D(32, 3, padding="same", activation="relu"))
    # model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(12, activation="softmax"))

    return model