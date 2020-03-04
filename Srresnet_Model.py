# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 10:35
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Srresnet_Model.py
import tensorflow.contrib as tc
from ops import *
import sys


# the shortcut of generator
def residual_block(inputs, output_channel, stride, scope, Resnet_weight, train=True):
    with tf.variable_scope(scope):
        net = conv2(inputs, 3, output_channel, stride, scope='conv_1')
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)

        net = conv2(net, 3, output_channel, stride, scope='conv_2')
        net = tf.layers.batch_normalization(net, training=train)
        net = Resnet_weight * net + inputs

    return net


def Generator_srresnet(z, y, num_resblock, Resnet_weight, reuse=False, train=True):
    img_size = 96
    s8 = int(img_size/8)

    with tf.variable_scope('g_net') as scope:
        if reuse:
            scope.reuse_variables()

        noise_vector = tf.concat([z, y], axis=1)

        #
        net_h0 = tc.layers.fully_connected(
            noise_vector, img_size * s8 * s8,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=None
        )

        net_h0 = tf.layers.batch_normalization(net_h0, training=train)
        net_h0 = tf.reshape(net_h0, [-1, s8, s8, img_size])
        net = tf.nn.relu(net_h0)

        # input_stage is the input of resnet network
        input_stage = net

        for i in range(1, num_resblock + 1, 1):
            name_scope = 'resblock_%d' % i
            net = residual_block(net, img_size, 1, name_scope, Resnet_weight,train=train)
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)

        #
        net = input_stage + net

        # pixelShuffler is used to change H * W into rH * rW
        net = conv2(net, 3, 256, 1, scope='conv1')
        net = pixelShuffler(net, scale=2)
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)

        net = conv2(net, 3, 256, 1, scope='conv2')
        net = pixelShuffler(net, scale=2)
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)

        net = conv2(net, 3, 256, 1, scope='conv3')
        net = pixelShuffler(net, scale=2)
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)

        # 模型最后的输出部分
        net = conv2(net, 9, 3, 1, scope='conv4')
        net = tf.nn.tanh(net)

        return net


def discriminator_block(inputs, output_channel, kernel_size, stride, scope, train=False):
    res = inputs

    with tf.variable_scope(scope):
        net = conv2(inputs, kernel_size, output_channel, stride, scope='dis_conv_1')
        net = tf.layers.batch_normalization(net, training=train)
        net = lrelu(net, 0.2)

        net = conv2(net, kernel_size, output_channel, stride, scope='dis_conv_2')
        net = tf.layers.batch_normalization(net, training=train)
        net = net + res

        net = lrelu(net, 0.2)
    return net


def Discriminator_srresnet(dis_inputs, reuse=False, train=True, dataset='anime'):
    with tf.variable_scope('d_net') as scope:
        if reuse:
            scope.reuse_variables()

        with tf.variable_scope('input_stage'):
            net = conv2(dis_inputs, 4, 32, 2, scope='conv')
            net = tf.layers.batch_normalization(net, training=train)
            net = lrelu(net, 0.2)

        net = discriminator_block(net, 32, 3, 1, 'disblock_1', train=train)
        net = discriminator_block(net, 32, 3, 1, 'disblock_1_1', train=train)

        net = conv2(net, 4, 64, 2, scope='dis_conv_1')

        # 识别器中的第二层卷积块
        net = discriminator_block(net, 64, 3, 1, 'disblock_2_1', train=train)
        net = discriminator_block(net, 64, 3, 1, 'disblock_2_2', train=train)
        net = discriminator_block(net, 64, 3, 1, 'disblock_2_3', train=train)
        net = discriminator_block(net, 64, 3, 1, 'disblock_2_4', train=train)

        net = conv2(net, 4, 128, 2, scope='dis_conv_2')
        net = lrelu(net, 0.2)

        # 识别器中第三层卷积块
        net = discriminator_block(net, 128, 3, 1, 'disblock_3_1', train=train)
        net = discriminator_block(net, 128, 3, 1, 'disblock_3_2', train=train)
        net = discriminator_block(net, 128, 3, 1, 'disblock_3_3', train=train)
        net = discriminator_block(net, 128, 3, 1, 'disblock_3_4', train=train)

        net = conv2(net, 3, 256, 2, scope='dis_conv_3')
        net = lrelu(net, 0.2)

        # 第四层卷积块
        net = discriminator_block(net, 256, 3, 1, 'disblock_4_1', train=train)
        net = discriminator_block(net, 256, 3, 1, 'disblock_4_2', train=train)
        net = discriminator_block(net, 256, 3, 1, 'disblock_4_3', train=train)
        net = discriminator_block(net, 256, 3, 1, 'disblock_4_4', train=train)

        net = conv2(net, 3, 512, 2, scope='dis_conv_4')
        net = lrelu(net, 0.2)

        net = discriminator_block(net, 512, 3, 1, 'disblock_5_1', train=train)
        net = discriminator_block(net, 512, 3, 1, 'disblock_5_2', train=train)
        net = discriminator_block(net, 512, 3, 1, 'disblock_5_3', train=train)
        net = discriminator_block(net, 512, 3, 1, 'disblock_5_4', train=train)

        net = conv2(net, 3, 1024, 2, scope='dis_conv_5')
        net = lrelu(net, 0.2)

        net = tf.reshape(net, [-1, 2 * 2 * 1024])
        print('the last layer of discriminator:', net)

        if dataset == 'anime':
            with tf.variable_scope('dense_layer_1'):
                net_class = fully_conneted(net, 23)

            with tf.variable_scope('dense_layer_2'):
                net = fully_conneted(net, 1)

        elif dataset == 'celebA':
            with tf.variable_scope('dense_layer_1'):
                net_class = fully_conneted(net, 1)

            with tf.variable_scope('dense_layer_2'):
                net = fully_conneted(net, 1)
        else:
            print('just support anime and celebA')
            sys.exit()
    return net, net_class
