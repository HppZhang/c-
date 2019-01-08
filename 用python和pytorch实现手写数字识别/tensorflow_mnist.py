#-*-coding:utf-8-*-
import tensorflow as tf
import image_extract as im
import numpy as np
image_batch = im.Dataset()
accuaracy_all = 0
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def read_input_feature(images):
    #print(np.shape(images))
    re_images = np.resize(images, (np.shape(images)[0], 784))
    #print(np.shape(re_images))
    return re_images
def weight_variable(shapes):
    initial = tf.truncated_normal(shapes, stddev=0.1)#或weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
def data_prepossing():
    x_train, y_train, x_validation, y_validation, x_test, y_test = image_batch.dataset()
    x_train = x_train.astype(np.float32)
    x_train = np.multiply(x_train, 1.0 / 255.0)
    x_validation = x_validation.astype(np.float32)
    x_validation = np.multiply(x_validation, 1.0 / 255.0)
    x_test = x_test.astype(np.float32)
    x_test = np.multiply(x_test, 1.0 / 255.0)
    x_train = read_input_feature(x_train)
    y_validation = np.asarray(y_validation, dtype=np.uint8)
    y_validation = dense_to_one_hot(y_validation, 10)
    x_validation = read_input_feature(x_validation)
    y_train = np.asarray(y_train, dtype=np.uint8)
    y_train = dense_to_one_hot(y_train, 10)
    y_test = np.asarray(y_test, dtype=np.uint8)
    y_test = dense_to_one_hot(y_test, 10)
    x_test = read_input_feature(x_test)
    return x_train, y_train, x_validation, y_validation, x_test, y_test

#sess=tf.InteractiveSession()#或将其用with tf.InteractiveSession() as sess:替换
x=tf.placeholder("float", shape=[None, 784])
y_=tf.placeholder("float", shape=[None, 10])
w_conv1=weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 =weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = weight_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuaracy_rate = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    x_train, y_train, x_validation, y_validation, x_test, y_test = data_prepossing()
    batch = 50
    for i in range(20):
        start = 0
        label = True
        perm0 = np.arange(len(x_train))
        np.random.shuffle(perm0)
        x_train = x_train[perm0]
        y_train = y_train[perm0]
        while (label):
            batch_x, batch_y, label = im.next_batch(x_train, y_train, start, batch)
            start += 1
            if start%100 == 0:
                train_accuracy = accuaracy_rate.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob:1.0})
                print("step%d, traing accuracy %g"%(start, train_accuracy))
            train_step.run(feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
        loss = cross_entropy.eval(feed_dict={x:x_validation, y_:y_validation, keep_prob:1.0})
        val_accuaracy = accuaracy_rate.eval(feed_dict={x:x_validation, y_:y_validation, keep_prob:1.0})
        print("epoch%d, validation loss %g" % (i, loss))
        print("epoch%d, validation accuracy %g" % (i, val_accuaracy))
        if val_accuaracy>accuaracy_all:
            accuaracy_all = val_accuaracy
            save_path = saver.save(sess, "./model")
            print("Model saved in files: ", save_path)
    saver.restore(sess, "./model.ckpt")
    print("Model restores.")
    test_loss = cross_entropy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    test_accuaracy = accuaracy_rate.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    print("epoch%d, test loss %g" % (i, test_loss))
    print("epoch%d, test accuracy %g" % (i, test_accuaracy))





