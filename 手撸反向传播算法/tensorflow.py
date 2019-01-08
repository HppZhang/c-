# 1 导入相关包
from time import time              # 计算训练模型总时间
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 2 加载mnist数据
start_time = time()
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 3 定义模型参数（权重、偏差）及占位符
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 4 计算y的预测值，定义y标签值的占位符
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)
y_label = tf.placeholder(tf.float32, [None, 10])

# 5 定义交叉熵损失，选择梯度下降优化方法
cross_entropy  = tf.reduce_mean(-tf.reduce_sum( y_label * tf.log(y_predict), axis=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_label: batch_ys})
    correct_predict = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype='float'))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
    print('模型训练总耗时：%.4f' %(time() - start_time)+'秒')
