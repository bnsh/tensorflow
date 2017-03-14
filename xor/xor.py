#! /usr/bin/python

import sys
import os
import tensorflow as tf
import numpy as np

def main():
	hiddensz = 3

	X = tf.placeholder(tf.float32, shape=[None,2], name="X")
	Y = tf.placeholder(tf.float32, shape=[None,1], name="Y")

	w1 = tf.Variable(tf.random_uniform([2,hiddensz],-1,1,dtype=tf.float32), name="w1")
	b1 = tf.Variable(tf.zeros([1,hiddensz],dtype=tf.float32), name="b1")

	w2 = tf.Variable(tf.random_uniform([hiddensz,1],-1,1,dtype=tf.float32), name="w2")
	b2 = tf.Variable(tf.zeros([1,1],dtype=tf.float32), name="b2")

	h = tf.nn.sigmoid(tf.matmul(X,w1) + b1)
	z = tf.matmul(h,w2) + b2

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=Y))

	opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

	X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
	Y_train = np.array([[0],[1],[1],[0]])

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	for i in xrange(0,1000):
		_, l = sess.run([opt, loss], feed_dict={X:X_train, Y:Y_train})
	out, err = sess.run([z,loss], feed_dict={X:X_train, Y:Y_train})

	print out, err

	
if __name__ == "__main__":
	main()
