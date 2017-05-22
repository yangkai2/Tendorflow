# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:31:51 2017

@author: 凯宾斯基
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
import tensorflow as tf
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])
#权重weight,偏置bias
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#定义分类模型
y=tf.nn.softmax(tf.matmul(x,W)+b)
#定义损失函数
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#定义优化算法
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
#开始训练迭代
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#评价准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))

#对模型进行保存
import tensorflow as tf

# 声明两个变量
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
init_op = tf.global_variables_initializer() # 初始化全部变量
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    sess.run(init_op)
    print("v1:", sess.run(v1)) # 打印v1、v2的值一会读取之后对比
    print("v2:", sess.run(v2))
    saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
    print("Model saved in file:", saver_path)