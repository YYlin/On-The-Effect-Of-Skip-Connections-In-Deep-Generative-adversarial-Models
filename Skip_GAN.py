# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 10:34
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Skip_GAN.py
from Dataload import load_anime_old, save_images, load_CelebA
from Srresnet_Model import Generator_srresnet, Discriminator_srresnet
import tensorflow as tf
import numpy as np
import sys


class Skip_GAN(object):
    def __init__(self, sess, epoch, batch_size, dataset_name, result_dir, z_dim, y_dim, checkpoint_dir, num_resblock,
                 Cycle_lr, Class_weight, Resnet_weight):
        self.sess = sess
        self.dataset_name = dataset_name
        self.result_dir = result_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.checkpoint_dir = checkpoint_dir
        self.num_resblock = num_resblock
        self.Cycle_lr = Cycle_lr
        self.Class_weight = Class_weight

        # La is used to increase the weight of image authenticity
        self.la = 10
        self.learningRateD = 2e-4
        self.learningRateG = 2e-4

        #
        self.Resnet_weight = Resnet_weight

        # 加载不同的数据集
        if self.dataset_name == 'anime':
            print('loading  anime .............')
            self.height = 96
            self.width = 96
            self.c_dim = 3

            self.data_X, self.data_Y = load_anime_old()
            print('self.data_X:', self.data_X.shape, 'self.data_y:', self.data_Y.shape)

        elif self.dataset_name == 'celebA':
            print('loading celebA  ...............')
            self.height = 96
            self.width = 96
            self.c_dim = 3

            self.data_X, self.data_Y = load_CelebA()
            print('self.data_X:', self.data_X.shape, 'self.data_y:', self.data_Y.shape)
        else:
            print('Sorry there is no option for ', self.dataset_name)
            sys.exit()

    def build_model(self):
        # some placeholder in our model
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
        self.img = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, 3], name='img')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])

        self.G_sample = Generator_srresnet(self.z, self.y, self.num_resblock, self.Resnet_weight)
        print('The return of Generator:', self.G_sample)

        # 识别器对真实图像进行判断
        D_real, C_real = Discriminator_srresnet(self.img, dataset=self.dataset_name)
        print('The return of Discriminator:', D_real, C_real)

        # 识别器对生成图像进行判断
        D_fake, C_fake = Discriminator_srresnet(self.G_sample, dataset=self.dataset_name, reuse=True)
        print('The return of Discriminator:', D_fake, C_fake)

        # 判断图像的类别
        self.C_real_loss = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_real, labels=self.y), axis=1))
        self.C_fake_loss = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_fake, labels=self.y), axis=1))

        # D_Loss 希望真实图像被判断为1 希望生成图像被判断为0
        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

        '''注意 la也即是我是用动态学习率的时候要关注的参数 
        但是我的目标是使得类别损失变得更加的大 而不是真伪的损失'''
        D_loss = D_real_loss + D_fake_loss
        self.DC_loss = (self.la * D_loss + self.C_real_loss)

        # 对生成模型的损失也在关注该模型
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        self.GC_loss = (self.la * G_loss + self.C_fake_loss)

        print('Calualtion the loss of Optimizer')
        self.theta_D = [v for v in tf.global_variables() if 'd_net' in v.name]
        self.theta_G = [v for v in tf.global_variables() if 'g_net' in v.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_updates = tf.train.AdamOptimizer(self.learningRateD, beta1=0.5, beta2=0.9).minimize(self.DC_loss,
                                                                                                       var_list=self.theta_D)
            self.g_updates = tf.train.AdamOptimizer(self.learningRateG, beta1=0.5, beta2=0.9).minimize(self.GC_loss,
                                                                                                       var_list=self.theta_G)
        self.sampler = Generator_srresnet(self.y, self.z, self.num_resblock, self.Resnet_weight, reuse=True, train=False)

    def train(self):
        print('begin training ...........')
        tf.global_variables_initializer().run()

        # sample_num 用于控制存储图像
        sample_num = 64
        tot_num_samples = min(sample_num, self.batch_size)
        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

        # 定义随机噪音以及标签 2019/09/29
        self.sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
        self.sample_y = self.data_Y[0:self.batch_size]

        counter = 0

        # shuffle the dataset 2019/9/29
        batch_offset = 0
        data_index = np.arange(self.data_X.shape[0])
        np.random.shuffle(data_index)
        self.data_X = self.data_X[data_index, :, :, :]
        self.data_Y = self.data_Y[data_index]

        # 这种方式会有使得小于batch_size个数据用不上
        for epoch in range(self.epoch):
            if batch_offset + self.batch_size > len(self.data_X):
                batch_offset = 0
                # shuffle dataset
                data_index = np.arange(self.data_X.shape[0])
                np.random.shuffle(data_index)
                self.data_X = self.data_X[data_index, :, :, :]
                self.data_Y = self.data_Y[data_index]
            else:
                # 首先是得到输入的数据
                batch_images = self.data_X[batch_offset:batch_offset + self.batch_size]
                batch_codes = self.data_Y[batch_offset:batch_offset + self.batch_size]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # 然后更新识别器
                for i_d_loss in range(3):
                    _, d_loss = self.sess.run([self.d_updates, self.DC_loss], feed_dict={self.img: batch_images,
                                                                                         self.y: batch_codes,
                                                                                         self.z: batch_z})
                for i_g_loss in range(1):
                    # 最后更新生成器模型
                    _, g_loss, _ = self.sess.run([self.g_updates, self.GC_loss, self.G_sample],
                                                 feed_dict={self.y: batch_codes, self.img: batch_images, self.z: batch_z})

                batch_offset = batch_offset + self.batch_size

                # display the loss every 10 steps
                if (counter % 10) == 0:
                    print('Epoch: %2d counter: %5d  d_loss: %.8f, g_loss: %.8f' % (epoch, counter, d_loss, g_loss))

                # save image every 500 steps
                if counter % 500 == 0:
                    samples = self.sess.run(self.sampler,
                                            feed_dict={self.z: self.sample, self.y: self.sample_y})

                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                self.result_dir + '/{}.png'.format(str(counter).zfill(7)))

                # save the model every 1000 steps
                if counter % 1000 == 0:
                    saver = tf.train.Saver(max_to_keep=5)
                    saver.save(self.sess, self.checkpoint_dir + '/{}'.format(str(counter).zfill(7)))

                if (counter % 100) == 0:
                    if self.Cycle_lr:
                        self.learningRateD = self.learningRateD * 0.99
                    if self.learningRateD < 0.0001:
                        self.learningRateD = 2e-4

                if (counter % 500) == 0:
                    if self.Class_weight:
                        if self.la > 25:
                            self.la = 25
                        else:
                            self.la = self.la * 1.5

                counter += 1
