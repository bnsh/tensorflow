#! /usr/bin/python

import sys
import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import conj

def simulate(X, W, b, stm1):
	# First, augment X with stm1
	concat = np.concatenate([X, stm1], 1)
	mul = np.dot(concat, W)
	brepeated = np.tile(b, (mul.shape[0], 1))
	res = np.tanh(mul + brepeated)
	return res, res

def main(argv):
	if os.path.exists("/tmp/wtf"):
		shutil.rmtree("/tmp/wtf")
	tf.set_random_seed(12345)
	with tf.Session() as sess:
		# np_data = np.array([[0,0],[0,1],[1,0],[1,1]])
		np_data = np.random.rand(4,23)
		with tf.name_scope("input") as scope:
			X_ = tf.placeholder(name="X", dtype=tf.float32, shape=(4, 23))
			Y_ = tf.placeholder(name="Y", dtype=tf.float32, shape=(4, 1))

		with tf.variable_scope("rnn") as scope:
			rnn = tf.contrib.rnn.BasicRNNCell(1)
			initial_state = rnn.zero_state(4, tf.float32)
			round1, state1 = rnn(X_, initial_state)
		with tf.variable_scope("rnn", reuse=True) as scope:
			round2, state2 = rnn(X_, state1)
			with tf.variable_scope("basic_rnn_cell", reuse=True):
				weights = tf.get_variable("weights")
				biases = tf.get_variable("biases")

		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter("/tmp/wtf", sess.graph)
		np_weights, np_biases, np_initial_state, tf_round1, tf_state1, tf_round2, tf_state2 = sess.run([weights, biases, initial_state, round1, state1, round2, state2], feed_dict={ X_: np_data, Y_: [[0],[1],[1],[0]] })

		np_round1, np_state1 = simulate(np_data, np_weights, np_biases, np_initial_state)
		np_round2, np_state2 = simulate(np_data, np_weights, np_biases, np_state1)
		print(np_round1)
		print(tf_round1)
		print(np_round2)
		print(tf_round2)

		writer.close()

if __name__ == "__main__":
	main(sys.argv[1:])
