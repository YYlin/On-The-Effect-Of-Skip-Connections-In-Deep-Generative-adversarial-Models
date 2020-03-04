# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 10:32
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : main.py
import os, sys
import tensorflow as tf
import argparse
from Skip_GAN import Skip_GAN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gan_type', type=str, default='Skip_GAN', choices=['Skip_GAN', 'DCGAN'],
                        help='The type of GAN')

    parser.add_argument('--dataset', type=str, default='anime', choices=['mnist', 'fashion-mnist', 'celebA', 'cifar10'],
                        help='The name of dataset')

    # 模型中训练数据一些信息
    parser.add_argument('--epoch', type=int, default=100000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--y_dim', type=int, default=23, help='class of images')
    parser.add_argument('--z_dim', type=int, default=100, help='The size of noise dimension')

    # some parameter about my paper
    parser.add_argument('--num_resblock', type=int, default=16, help='The number of shortcut in generator')
    parser.add_argument('--Cycle_lr', type=bool, default=True, help='Use Cycle learningRate for The model')
    parser.add_argument('--Class_weight', type=bool, default=True, help='Use Gradually increasing Class weight')

    parser.add_argument('--Resnet_weight', type=float, default=10.0, help='The weight of ResNet')

    # some folder to save file
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')

    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Directory name to save the model')

    return check_args(parser.parse_args())


def check_args(args):
    args.result_dir = args.dataset + '_' + args.gan_type + '_batch_size_' + str(args.batch_size) + '_epoch_' + str(args.epoch) \
                      + '_' + args.result_dir + '_num_resblock_' + str(args.num_resblock) + '_Cycle_lr_' \
                      + str(args.Cycle_lr) + '_Class_weight_' + str(args.Class_weight)

    if os.path.exists(args.result_dir):
        print('are you sure? the result folder is exist')
    else:
        os.makedirs(args.result_dir)

    args.checkpoint_dir = args.dataset + '_' + args.gan_type + '_batch_size_' + str(args.batch_size) + '_epoch_' + str(
        args.epoch) + '_' + args.checkpoint_dir + '_num_resblock_' + str(args.num_resblock) + '_Cycle_lr_' + \
                          str(args.Cycle_lr) + '_Class_weight_' + str(args.Class_weight) +'/checkpoints'

    if os.path.exists(args.checkpoint_dir):
        print('are you sure?，checkpoint_dir is exist')
    else:
        os.makedirs(args.checkpoint_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def main():
    args = parse_args()
    if args is None:
        print('the args is None, check the parameter of input')
        sys.exit()
    else:
        print('args:', args)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if args.gan_type == 'Skip_GAN':
            gan = Skip_GAN(sess, epoch=args.epoch, batch_size=args.batch_size, dataset_name=args.dataset, result_dir=args.result_dir,
                             z_dim=args.z_dim, y_dim=args.y_dim, checkpoint_dir=args.checkpoint_dir, num_resblock=args.num_resblock,
                             Cycle_lr=args.Cycle_lr, Class_weight=args.Class_weight, Resnet_weight=args.Resnet_weight)
        else:
            raise Exception('[!] There is no option for ' + args.gan_type)

        print('[*] build model ............')
        gan.build_model()

        print('[*] training ................')
        gan.train()

        print('[*] Training finished!')


if __name__ == '__main__':
    main()

