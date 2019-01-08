# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import struct
import cv2
#import matplotlib.pyplot as plt

# 训练集文件
train_images_idx3_ubyte_file = '../手写数字数据集/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '../手写数字数据集/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '../手写数字数据集/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '../手写数字数据集/t10k-labels.idx1-ubyte'


class Dataset():
    def __init__(self, idx3_ubyte_file=None):
        self.validation_size = 5000
        self._index_in_epoch = 0
        self.idx3_ubyte_file = idx3_ubyte_file
        self._epochs_completed = 0
    def decode_idx3_ubyte(self, idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        print ('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((num_images, num_rows, num_cols))
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print ('已解析 %d' % (i + 1) + '张')
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            #images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            offset += struct.calcsize(fmt_image)
        return images

    def decode_idx1_ubyte(self, idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print ('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print ('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels


    def load_train_images(self, idx_ubyte_file=train_images_idx3_ubyte_file):
        """
        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
        return self.decode_idx3_ubyte(idx_ubyte_file)


    def load_train_labels(self, idx_ubyte_file=train_labels_idx1_ubyte_file):
        """
        TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.

        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
        return self.decode_idx1_ubyte(idx_ubyte_file)


    def load_test_images(self, idx_ubyte_file=test_images_idx3_ubyte_file):
        """
        TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  10000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
        return self.decode_idx3_ubyte(idx_ubyte_file)


    def load_test_labels(self,idx_ubyte_file=test_labels_idx1_ubyte_file):
        """
        TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  10000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.

        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
        return self.decode_idx1_ubyte(idx_ubyte_file)

    def dataset(self, validation_size=5000):
        train_labels= self.load_train_labels()
        train_images = self.load_train_images()
        perm0 = np.arange(len(train_images))
        np.random.shuffle(perm0)
        train_images = train_images[perm0]
        train_labels = train_labels[perm0]
        self.validation_size=validation_size
        if not 0 <= self.validation_size <= len(train_images):
            raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                             .format(len(train_images), validation_size))
        self.validation_image = train_images[:validation_size]
        self.validation_label = train_labels[:validation_size]
        self.train_images = train_images[validation_size:]
        self.train_labels = train_labels[validation_size:]
        self.test_images = self.load_test_images()
        self.test_label = self.load_test_labels()
        return self.train_images, self.train_labels, self.validation_image, self.validation_label, self.test_images, self.test_label

def next_batch(images, labels, itertor, batch):
    if itertor*batch+batch <= np.shape(images)[0]:
        re_images = images[itertor*batch:itertor*batch+batch]
        re_label = labels[itertor*batch:itertor*batch+batch]
        return re_images, re_label, True
    if itertor*batch+batch > np.shape(images)[0]:
        return images[itertor*batch:], labels[itertor*batch:], False


"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: mnist_decoder.py
@time: 2016/8/16 20:03

对MNIST手写数字数据文件转换为bmp图片文件格式。
数据集下载地址为http://yann.lecun.com/exdb/mnist。
相关格式转换见官网以及代码注释。

========================
关于IDX文件格式的解析规则：
========================
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""
