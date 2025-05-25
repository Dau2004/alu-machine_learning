#!/usr/bin/env python3
"""Class Neuron that defines a single neuron performing binary classification
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Function that returns two placeholders, x and y, for the neural network

    Args:
        nx (int): the number of feature columns in our data
        classes (int): the number of classes in our classifier

    Returns:
        x (tf.Tensor): the placeholder for the input data to the neural network
        y (tf.Tensor): the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(dtype="float", shape=[None, nx], name="x")
    y = tf.placeholder(dtype="float", shape=[None, classes], name="y")
    return x, y