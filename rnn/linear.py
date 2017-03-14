#! /usr/bin/python

import sys
import os
import random
import tensorflow as tf

"""
	This is a demo of tensorflow's ... flow, I guess.
	It's just a simple linear algebra solver.
"""

def main():
# First, let's construct our data.
# We'll generate our data. We'll make 2 dimensional
# data: so, data that takes (a, b) and outputs y
# according to wa * a + wb * b + wbias + gaussian(m, sd) = y
# We'll construct sz points

	sz = 16384

	m = random.uniform(-10,10)
	sd = random.uniform(1,5)

	wbias = random.uniform(-10,10)
	wa = random.uniform(-10,10)
	wb = random.uniform(-10,10)

	data = [(random.gauss(0,1), random.gauss(0,1)) for _ in range(0,sz)]
	targets = [[wa * a + wb * b + wbias + random.gauss(m, sd)] for a,b in data]

	# We need to make two placeholders: one for our data, and another for our targets
	data_ = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="data")
	targets_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")

	# The next three lines make the "flow graph" for the calculation
	weights = tf.Variable(tf.random_uniform((2, 1), -1, 1, tf.float32), name="weights")
	bias = tf.Variable(tf.random_uniform((1,1), -1, 1, tf.float32), name="bias")
	output = tf.matmul(data, weights) + bias

	# Now, the "flow graph" for the output.
	err = tf.subtract(targets, output, name="error")
	squared_err = tf.pow(err, 2)
	mean_squared_error = tf.reduce_mean(squared_err, name="mean_squared_error")

	# This is the optimizer.
	opt = tf.train.GradientDescentOptimizer(learning_rate=0.1, name="optimizer").minimize(mean_squared_error)

	session = tf.Session()

	# This generates the pretty graph tensorflow is adored the world over for.
	# tensorboard --logdir=/tmp/graphs
	writer = tf.summary.FileWriter('/tmp/graphs', session.graph)

	session.run(tf.global_variables_initializer())

	feed_dict={
		data_: data,
		targets_: targets
	}

	starting_mse = session.run(mean_squared_error, feed_dict=feed_dict)
	for epoch in range(0, 1024):
		session.run(opt, feed_dict=feed_dict)
	ending_mse, w, b  = session.run([mean_squared_error, weights, bias], feed_dict=feed_dict)
	print("Started mse=%.7f" % (starting_mse))
	print("  Ended mse=%.7f" % (ending_mse))
	print("Ground Truth:")
	print("	y = (%.7f) a + (%.7f) b + (%.7f) + N(%.7f, %.7f)" % (wa, wb, wbias, m, sd))
	print("Inferred:")
	print("	y = (%.7f) a + (%.7f) b + (%.7f)" % (w[0], w[1], b))

	writer.close()
	session.close()


if __name__ == "__main__":
	main()
