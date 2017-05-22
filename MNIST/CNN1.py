# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:29:04 2017

@author: 凯宾斯基
"""
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()
#定义权重,，偏置初始化函数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#定义卷积层和池化层函数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],
                          padding='SAME')
#定义输入(即第一层)
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
#第一个卷积层
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#第二个卷积层
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#全连接层
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#用Dropout方法减轻过拟合
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#最后连接一个分类层
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#定义损失函数
loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),
                                   reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
#评价准确率
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.global_variables_initializer().run()
def every_accuracy(a,b):
    train_accuracy=accuracy.eval(feed_dict={x:a,y:b,
                                                keep_prob:1.0})
    print('step %d,training acuracy %g'%(i,train_accuracy))
for i in range(1000):
    batch=mnist.train.next_batch(100)
    if i%100==0:
        every_accuracy(batch[0],batch[1])
    train_step.run(feed_dict={x:batch[0],y:batch[1],keep_prob:0.5})
print('test accuracy %g'%accuracy.eval(feed_dict={x:mnist.test.images[0:1000,:],
                                                  y:mnist.test.labels[0:1000,:],
                                                  keep_prob:1.0}))
#保存模型
saver=tf.train.Saver()
#print('W:',sess.run(W))
#print('b:',sess.run(b))
save_path=saver.save(sess,"CNN1/save_net.ckpt")
print('save to path:',save_path)