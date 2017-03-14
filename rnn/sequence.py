#! /usr/bin/python

import sys
import os
import shutil
import random
import tensorflow as tf
import numpy as np

batch_sz = 1024
sequence_sz = 8
width = 2
depth = 3
lr = 0.0001

def oddbefore(accumulator, x):
	if len(accumulator) == 0:
		ob = 0
	else:
		ob = accumulator[-1]

	v = 0 if ob == x else 1

	return accumulator + [v]

def generate_data(batchsz, sequencesz):
	# Generate both inputs and targets.
	data = []
	target = []
	for q in xrange(0, batchsz):
		data.append([random.randint(0,1) for _ in xrange(0, sequencesz)])
		target.append(reduce(oddbefore, data[q], []))
	target = map(lambda x: [x[-1]], target)
	return data, target

def main(argv):
	if os.path.exists("/tmp/wtf"):
		shutil.rmtree("/tmp/wtf")
	tf.set_random_seed(12345)
	with tf.Session() as sess:
		with tf.variable_scope("input") as scope:
			X_ = tf.placeholder(name="X", dtype=tf.float32, shape=(None, sequence_sz))
		with tf.variable_scope("target") as scope:
			Y_ = tf.placeholder(name="Y", dtype=tf.float32, shape=(None, 1))

		for i in xrange(0, sequence_sz):
			with tf.variable_scope("input", reuse=True) as scope:
				piece = tf.slice(X_, [0, i], [-1,1], name="slice_%d" % (i))
			with tf.variable_scope("rnn", reuse=None if (i == 0) else True) as scope:
				rnn = tf.contrib.rnn.MultiRNNCell(
					[tf.contrib.rnn.BasicRNNCell(width) for q in xrange(0, depth)]
				)
				if i == 0:
					state = initial_state = rnn.zero_state(batch_sz, tf.float32)
				output, state = rnn(piece, state)

		with tf.variable_scope("fc") as scope:
			W = tf.get_variable("W", [width, 1], dtype=tf.float32)
			b = tf.get_variable("b", [1, 1], dtype=tf.float32)
			final = tf.nn.sigmoid(tf.matmul(output, W) + b)

		with tf.variable_scope("loss") as scope:
			cost = -tf.reduce_mean(tf.multiply(Y_, tf.log(final)) + tf.multiply((tf.ones(final.shape) + tf.negative(Y_)), tf.log(tf.ones(final.shape) + tf.negative(final))))
			tf.summary.scalar("cost", cost)

		with tf.variable_scope("gradients") as scope:
			opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

		merged = tf.summary.merge_all()

		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter("/tmp/wtf", sess.graph)

		c = None
		epoch = 0
		while c is None or c > 0.01 or True:
			data, targets = generate_data(batch_sz, sequence_sz)
			np_data = np.array(data)
			np_targets = np.array(targets)
			_, _m,  c, f = sess.run([opt, merged, cost, final], feed_dict={X_: np_data, Y_: np_targets})
			epoch += 1
			writer.add_summary(_m, epoch)
			if (epoch % 1024) == 0:
				print c

		o, c = sess.run([final, cost], feed_dict={ X_: np_data, Y_: np_targets })


		writer.close()

if __name__ == "__main__":
	main(sys.argv[1:])
