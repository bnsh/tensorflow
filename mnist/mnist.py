#! /usr/bin/python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

"""
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
['__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_epochs_completed', '_images', '_index_in_epoch', '_labels', '_num_examples', 'epochs_completed', 'images', 'labels', 'next_batch', 'num_examples']
"""

def main():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


	Xtrain = mnist.train.images
	Ytrain = mnist.train.labels

	Xvalidation = mnist.validation.images
	Yvalidation = mnist.validation.labels

	Xtest = mnist.test.images
	Ytest = mnist.test.labels

	input_feature_sz = Xtrain.shape[1]
	output_feature_sz = Ytrain.shape[1]

	X_ = tf.placeholder(tf.float32, shape=[None, input_feature_sz], name="X")
	Y_ = tf.placeholder(tf.float32, shape=[None, output_feature_sz], name="Y")

	hiddensz = 256
	layers = 2
	activation = tf.nn.tanh

	weights = [None for i in xrange(0, layers)]
	biases = [None for i in xrange(0, layers)]

# I don't really know that I _need_ to store all these in arrays.
	z = [None for i in xrange(0, layers)]
	a = [None for i in xrange(0, layers)]
	final_z = None
	for i in xrange(0, layers):
		input = X_
		inputsz = hiddensz
		outputsz = hiddensz

		if i == 0:
			inputsz = input_feature_sz
		else:
			input = a[i-1]
		if i == (layers-1):
			outputsz = output_feature_sz

		weights[i] = tf.Variable(tf.random_uniform([inputsz, outputsz], -1, 1, dtype=tf.float32), name="weights_%02d" % (1+i))
		biases[i] = tf.Variable(tf.random_uniform([1, outputsz], -1, 1, dtype=tf.float32), name="biases_%02d" % (1+i))
		z[i] = tf.matmul(input, weights[i]) + biases[i]
		a[i] = activation(z[i])
		final_z = z[i]
		corrects = tf.equal(tf.argmax(a[i], 1), tf.argmax(Y_, 1))
		accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(final_z, Y_))
	opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
		

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	i = 0
	done = False
	for i in xrange(0, 100):
		for mb in xrange(0, 55):
			batch_x, batch_y = mnist.train.next_batch(1000)
			sess.run(opt, feed_dict={X_:batch_x, Y_:batch_y})
		ll, a = sess.run([loss, accuracy], feed_dict={X_:Xvalidation, Y_:Yvalidation})
		print "i=%d	loss=%.7f	accuracy=%.7f" % (i, ll, a)

if __name__ == "__main__":
	main()
