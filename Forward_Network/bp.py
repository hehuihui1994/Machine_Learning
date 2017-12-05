import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from point import Point


def sigmod(z):
	return 1.0/(1.0+tf.exp(-z))


def draw_line(theta):
	ax = np.linspace(20, 60, 1000)
	ay = (-theta[1]/theta[2])*ax-(theta[0]/theta[2])*100
	plt.plot(ax, ay)


p = Point()
p.add_ones()
x = tf.placeholder("float", [None, 3])
y = tf.placeholder("float", [None, 1])

weight_1 = tf.Variable(tf.random_uniform([3, 10], minval=-1.0, maxval=1.0, dtype=tf.float32, name="weight_1"))
biase_1 = tf.Variable(tf.random_uniform([1, 10], minval=-1.0, maxval=1.0, dtype=tf.float32, name="biase_1"))
middle_output = tf.nn.relu(tf.matmul(x, weight_1) + biase_1)

weight_2 = tf.Variable(tf.random_uniform([10, 1], minval=-1.0, maxval=1.0, dtype=tf.float32, name="weight_2"))
biase_2 = tf.Variable(tf.random_uniform([1, 1], minval=-1.0, maxval=1.0, dtype=tf.float32, name="biase_2"))
last_out = tf.matmul(middle_output, weight_2) + biase_2

cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - last_out), reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for time in range(1000):
		for i in range(80):
			print(sess.run(cost, feed_dict={x: p.onex[i].copy().reshape(1, 3), y: p.y[i].copy().reshape(1, 1)}))
	p.print_points()
	plt.show()
draw_line()