# -*- coding: utf-8 -*-
import numpy as np
import cv2
import xlrd
import h5py
import image_extract as im

image_batch = im.Dataset()

def get_label(label):
    shape = np.shape(label)
    re_label = np.zeros([shape[0], 10])
    #print(np.shape(re_label))
    for i in range(shape[0]):
        re_label[i][int(label[i])] = 1.0
    return re_label
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  print("offect",np.shape(index_offset))
  print("reval", np.shape(labels_dense.ravel()))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def read_input_feature(images):
    #print(np.shape(images))
    re_images = np.resize(images, (np.shape(images)[0], 784))
    #print(np.shape(re_images))
    return re_images
def layer(w,b,x):
    y = np.dot(x, w) + b
    t = -1.0*y
    y = 1.0/(1+np.exp(t))
    return y
def mytrain():
    x_train, y_train, x_validation, y_validation, x_test, y_test = image_batch.dataset()
    x_train = x_train.astype(np.float32)
    x_train = np.multiply(x_train, 1.0 / 255.0)
    x_train = read_input_feature(x_train)
    x_validation = x_validation.astype(np.float32)
    x_validation = np.multiply(x_validation, 1.0 / 255.0)
    x_test = x_test.astype(np.float32)
    x_test = np.multiply(x_test, 1.0 / 255.0)
    y_validation = np.asarray(y_validation, dtype=np.uint8)
    y_validation = dense_to_one_hot(y_validation, 10)
    x_validation = read_input_feature(x_validation)
    y_train = np.asarray(y_train, dtype=np.uint8)
    y_train = dense_to_one_hot(y_train, 10)
    y_test = np.asarray(y_test, dtype=np.uint8)
    y_test = dense_to_one_hot(y_test, 10)
    x_test = read_input_feature(x_test)
    ori_correct_rate = 0
    epoch = 20#int(input("epoch"))
    learning_rate = 0.005#float(input("please input learning rate:"))
    inn = 784
    hid = 500#int(input("隐层神经元个数"))
    batch = 60#int(input("batch"))
    out = 10

    '''data = h5py.File('./weight.h5', 'r', driver='core')
    w=data['w']
    w = np.mat(w)
    w1 = data['w1']
    w1 = np.mat(w1)
    b = data['b']
    b=np.mat(b)
    b1 = data['b1']
    b1 = np.mat(b1)
    data.close()'''
    w = np.random.randn(hid, out)
    w1 = np.random.randn(inn, hid)
    w = np.mat(w)
    w1 = np.mat(w1)
    b = np.mat(np.random.randn(1, out))
    b1 = np.mat(np.random.randn(1, hid))
    #print(np.shape(b1))
    for i in range(epoch):
        perm0 = np.arange(len(x_train))
        np.random.shuffle(perm0)
        x_train = x_train[perm0]
        y_train = y_train[perm0]
        start = 0
        label = True
        while(label):
            batch_x, batch_y, label = im.next_batch(x_train, y_train, start, batch)
            start+=1
            y = layer(w1, b1, batch_x)
            y1 = layer(w, b, y)
            y1 = np.asarray(y1)
            o_update = np.multiply(np.multiply((batch_y-y1), y1), (1-y1))
            h_update = np.multiply(np.multiply(np.dot(o_update, (w.T)), np.mat(y)), (1-y))
            outw_update = learning_rate*np.dot((y.T), o_update)
            outb_update = learning_rate*o_update
            hidw_update = learning_rate*np.dot((batch_x.T), h_update)
            hidb_update = learning_rate*h_update
            w = w+outw_update
            b = (b+outb_update)[0]
            w1 = w1+hidw_update
            b1 = (b1+hidb_update)[0]
        y_val = layer(w1, b1, x_validation)
        y1_val = layer(w, b, y_val)
        y1_val = np.asarray(y1_val)
        loss = 0.5 * np.sum(np.square(np.sum(y_validation - y1_val, 1)))
        correct = np.sum(np.equal(np.argmax(y_validation, 1), np.argmax(y1_val, 1)))
        all = len(x_validation)
        loss = loss/all
        correct_rate = float(correct)/all
        print("loss in validation is ", loss)
        print("correct_rate in validation is", correct_rate)
        if correct_rate>ori_correct_rate:
            ori_correct_rate = correct_rate
            data = h5py.File('./weight.h5', 'w')
            f = open("./validation_result.txt", "a")
            f.write("The best validation rate is "+ str(correct_rate))
            data.create_dataset("w", dtype='float64', data = w)
            data.create_dataset("w1", dtype='float64', data=w1)
            data.create_dataset("b", dtype='float64', data=b)
            data.create_dataset("b1", dtype='float64', data=b1)
            data.close()
            f.close()
    data = h5py.File('./weight.h5', 'r', driver='core')
    w=data['w']
    w = np.mat(w)
    w1 = data['w1']
    w1 = np.mat(w1)
    b = data['b']
    data.close()
    b=np.mat(b)
    b=b[0]
    b1 = data['b1']
    b1 = np.mat(b1)
    b1=b1[0]
    y_test = layer(w1, b1, x_test)
    y1_test = layer(w, b, y_test)
    loss = 0.5 * np.sum(np.square(np.sum(y_test - y1_test, 1)))
    correct = np.sum(np.equal(np.argmax(y_test, 1), np.argmax(y1_test, 1)))
    all = len(x_test)
    loss = loss / all
    correct_rate = float(correct) / all
    print("loss in test is ", loss)
    print("correct_rate in test is", correct_rate)
    f = open("./validation_test_result.txt", "a")
    f.write("The best validation rate is " + str(correct_rate))

if __name__ == '__main__':
    mytrain()