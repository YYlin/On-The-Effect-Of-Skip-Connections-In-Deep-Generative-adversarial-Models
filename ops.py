# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 10:37
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : ops.py
import tensorflow as tf


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        '''
        power iteration
        Usually iteration = 1 will be enough
        '''

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv2(batch_input, kernel=3, output_channel=64, stride=1, scope='conv'):
    with tf.variable_scope(scope):
        w = tf.get_variable('kernel', shape=[kernel, kernel, batch_input.get_shape()[-1], output_channel],
                            regularizer=None, initializer=tf.random_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input=batch_input, filter=spectral_norm(w), strides=[1, stride, stride, 1],
                            padding='SAME')
        return conv


def phaseShift(inputs, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])
    return tf.reshape(X, shape_2)


def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, shape_1, shape_2) for x in input_split], axis=3)
    return output


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fully_conneted(x, units, scope='fully_0'):
    with tf.variable_scope(scope):

        shape = x.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable('kernel', [channels, units], tf.float32, regularizer=None,
                            initializer=tf.random_normal_initializer(stddev=0.02))

        x = tf.matmul(x, spectral_norm(w))
        return x

