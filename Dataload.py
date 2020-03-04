# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 21:28
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Dataload.py
import csv
from scipy import misc
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 头发的颜色和眼睛的颜色 其类别分别是12 11
hair_color = ['aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair',
              'black hair', 'brown hair', 'blonde hair', 'orange hair', 'white hair']

eye_color = ['aqua eyes', 'gray eyes', 'green eyes', 'red eyes', 'purple eyes', 'pink eyes',
             'blue eyes', 'black eyes', 'brown eyes', 'orange eyes', 'yellow eyes']


# 按照dataload的方式 首先加载的是hair的颜色 然后是eye的颜色
def make_one_hot(hair, eye, all_tag=23):
    tag_vec = np.zeros(all_tag)
    tag_vec[hair] = 1
    tag_vec[eye + 12] = 1
    return tag_vec


# 对图像进行中心剪切操作
def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


# 该加载数据的方式和原始论文中的一致 主要用于对比自己 加载数据的效果
def load_anime_old():
    tag_path = 'data/tags_clean.csv'
    img_dir = 'data/faces'

    # img_feat:保存图像数据  attrib_tags:保存标签数据
    img_feat = []
    attrib_tags = []
    img_size = 96
    resieze = int(img_size * 1.15)

    with open(tag_path, 'r') as f:
        for idx, row in enumerate(csv.reader(f)):
            tags = row[1].split('\t')

            had_hair = False
            had_eye = False
            skip = False
            hair = 'unk'
            eyes = 'unk'

            for t in tags:
                if t != '':
                    tag = t.split(':')[0].strip()

                    # 眼睛颜色为双色的去掉
                    if tag == 'bicolored eyes':
                        skip = True
                        break

                    # 如果眼睛颜色在12类颜色之中的话 记录该颜色并标记数据中已经有了眼睛颜色了
                    if tag in eye_color:
                        if had_eye:
                            skip = True
                            break
                        eyes = tag
                        had_eye = True

                    if tag in hair_color:
                        if had_hair:
                            skip = True
                            break
                        hair = tag
                        had_hair = True

            # 每一行数据检查之后 只要有异常数据就跳过该图像
            if skip or hair == 'unk' or eyes == 'unk':
                continue

            hair_idx = hair_color.index(hair)
            eyes_idx = eye_color.index(eyes)

            '''
            # 测试的时候使用
            if idx > 100:
                break
            '''
            # 读取图像并且把数据resize成合适的size
            img_path = os.path.join(img_dir, '{}.jpg'.format(idx))
            feat = misc.imread(img_path)
            feat = misc.imresize(feat, [img_size, img_size, 3])
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat)

            # 对图像进行旋转操作
            m_feat = np.fliplr(feat)
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(m_feat)

            # 旋转10度 并进行中心裁剪操作
            feat_p10 = misc.imrotate(feat, 10)
            feat_p10 = misc.imresize(feat_p10, [resieze, resieze, 3])
            feat_p10 = crop_center(feat_p10, img_size, img_size)
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat_p10)

            # 旋转-10度 并进行中心裁剪操作
            feat_m5 = misc.imrotate(feat, -10)
            feat_m5 = misc.imresize(feat_m5, [resieze, resieze, 3])
            feat_m5 = crop_center(feat_m5, img_size, img_size)

            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat_m5)

    # 图像数据的取值范围取到(-1, 1)之间 / 127.5 - 1
    img_feat = np.array(img_feat, dtype='float32') / 127.5 - 1.

    # 直接返回测试对应的标签[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
    tag_vec = []
    for tmp in attrib_tags:
        tag_vec.append(make_one_hot(tmp[0], tmp[1]))
    tag_vec = np.array(tag_vec)

    return img_feat, tag_vec


def inverse_transform(images):
    return (images+1.)/2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


# save image
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


# 加载celebA数据集
def load_CelebA():
    target = 'Male'
    img_path = '../Dataset/celebA'

    attrib_tags = []
    img_feat = []

    img_size = 96
    resieze = int(img_size * 1.5)

    with open('../Dataset/text_data/list_attr_celeba.txt', 'r') as f:
        lines = f.readlines()
        all_tags = lines[1].strip('\n').split()
        print('确认一下all_tags的类型和具体的属性值', type(all_tags), all_tags)

        for i in range(2, len(lines)):
            line = lines[i].strip('\n').split()

            # 读取对应的标签数据
            if int(line[all_tags.index(target) + 1]) == 1:
                tmp_label = [1]
            elif int(line[all_tags.index(target) + 1]) == -1:
                tmp_label = [0]

            image_path = os.path.join(img_path, line[0])
            # print('image_path:', image_path, tmp_label)

            if i > 100:
                print('测试的时候仅仅使用100个数据')
                break

            # 读取数据并resize成合适的尺寸
            feat = misc.imread(image_path)
            feat = misc.imresize(feat, [img_size, img_size, 3])
            attrib_tags.append(tmp_label)
            img_feat.append(feat)

            # 对图像进行旋转操作
            m_feat = np.fliplr(feat)
            attrib_tags.append(tmp_label)
            img_feat.append(m_feat)

            # 旋转10度 并进行中心裁剪操作
            feat_p10 = misc.imrotate(feat, 10)
            feat_p10 = misc.imresize(feat_p10, [resieze, resieze, 3])
            feat_p10 = crop_center(feat_p10, img_size, img_size)
            attrib_tags.append(tmp_label)
            img_feat.append(feat_p10)

            # 旋转-10度 并进行中心裁剪操作
            feat_m5 = misc.imrotate(feat, -10)
            feat_m5 = misc.imresize(feat_m5, [resieze, resieze, 3])
            feat_m5 = crop_center(feat_m5, img_size, img_size)

            attrib_tags.append(tmp_label)
            img_feat.append(feat_m5)

    img_feat = np.array(img_feat, dtype='float32') / 127.5 - 1
    tags = np.array(attrib_tags)
    return img_feat, tags


# 加载测试集 用于anime数据集
def load_test(test_path):
    test = []
    with open(test_path, 'r') as f:

        for line in f.readlines():
            hair = 0
            eye = 0
            if line == '\n':
                break
            line = line.strip().split(',')[1]

            p = line.split(' ')
            p1 = ' '.join(p[:2]).strip()
            p2 = ' '.join(p[-2:]).strip()

            if p1 in hair_color:
                hair = hair_color.index(p1)
            elif p2 in hair_color:
                hair = hair_color.index(p2)

            if p1 in eye_color:
                eye = eye_color.index(p1)
            elif p2 in eye_color:
                eye = eye_color.index(p2)

            test.append(make_one_hot(hair, eye))

    return test

