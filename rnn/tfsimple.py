#! /usr/bin/python

import sys
import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

# Because I've precomputed the weights, unfortunately, this version is
# _constrained_ to use rnnwidth=2 see tfsimplen.py for
# making this truly parametrizable
rnnwidth = 2
batch_sz = 1024
sequence_sz = 8
lr=0.002
preinitialize=True
# tensorboard --logdir="/tmp/wtf"
tensorboarddir="/tmp/wtf"

def generate_data(batchsz, sequencesz):
	# Generate both inputs and targets.
	data = []
	target = []
	for q in range(0, batchsz):
		data.append([random.randint(0,1) for _ in range(0, sequencesz)])
	data = np.array(data)
	target = np.sum(data,1)
	target = np.mod(target,2).reshape(target.shape[0],1)
	return data, target

def spigot(x):
	if preinitialize:
		scale = 1 # changing this to 0.125 will still work, and learn. But, 0.0625 will _not_! (for sequence_sz=8)
		return tf.constant_initializer(scale * np.array(x))
	else:
		return None

def storethese(hr):
	for key, value in hr.items():
		tf.add_to_collection(key, value)

def main():
	if os.path.exists(tensorboarddir):
		shutil.rmtree(tensorboarddir)
	# Interestingly, 12346 works with sequence_sz=2. 12345 fails with sequence_sz=2
	# This bears looking into.
	tf.set_random_seed(12345)

	with tf.Session() as sess:
		with tf.variable_scope("input"):
			x_ = tf.placeholder(dtype=tf.float32, shape=(None, sequence_sz), name="x")

		with tf.variable_scope("target"):
			y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

		with tf.variable_scope("rnn", reuse=None) as scope:
			with tf.variable_scope("basic_rnn_cell", reuse=None):
				weights = tf.get_variable("weights", [3, rnnwidth], dtype=tf.float32, initializer=spigot([[-10,-10],[5,5],[-5,-5]]))
				biases = tf.get_variable("biases", [rnnwidth], dtype=tf.float32, initializer=spigot([5,15]))

		for i in range(0, sequence_sz):
			with tf.variable_scope("input", reuse=True) as scope:
				piece = tf.slice(x_, [0, i], [-1,1], name="slice_%d" % (i))

			reuse = None if (i==0) else True
			with tf.variable_scope("rnn", reuse=reuse) as scope:
				rnn = tf.contrib.rnn.BasicRNNCell(rnnwidth)
				if i == 0:
					state = initial_state = rnn.zero_state(batch_sz, tf.float32)
				with tf.variable_scope("basic_rnn_cell", reuse=True) as scope:
					output, state = rnn(piece, state, scope=scope)

		with tf.variable_scope("fc") as scope:
			W = tf.get_variable("W", [rnnwidth, 1], dtype=tf.float32, initializer=spigot([[-5],[5]]))
			b = tf.get_variable("b", [1, 1], dtype=tf.float32, initializer=spigot([[-5]]))
			final = tf.nn.sigmoid(tf.matmul(output, W) + b)

		with tf.variable_scope("loss") as scope:
			cost = -tf.reduce_mean(tf.multiply(y_, tf.log(final)) + tf.multiply((tf.ones(final.shape) + tf.negative(y_)), tf.log(tf.ones(final.shape) + tf.negative(final))))
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater_equal(final, 0.5), dtype=tf.float32), y_), dtype=tf.float32))
			tf.summary.scalar("cost", cost)
			tf.summary.scalar("accuracy", accuracy)

		with tf.variable_scope("gradients") as scope:
			adam = tf.train.AdamOptimizer(learning_rate=lr)
			grad_vars = adam.compute_gradients(cost)
			opt = adam.apply_gradients(grad_vars)

			gradients = reduce(lambda acc, v: tf.concat([acc, v], axis=0), map(lambda x: tf.reshape(x[0], [reduce(lambda acc, p: acc * (p.value), x[0].shape, 1)]), grad_vars))
			tf.summary.histogram("gradients", gradients)

		merged = tf.summary.merge_all()

		hstate = np.array([[0],[0]])

		writer = tf.summary.FileWriter(tensorboarddir, sess.graph)
		storethese({
			"x_": x_,
			"y_": y_,
			"cost": cost,
			"accuracy": accuracy,
			"final": final
		})
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		best_cost = None
		epoch = 0
		while True:
			np_data, np_targets = generate_data(batch_sz, sequence_sz)
			_, _m,  c, a, f = sess.run([opt, merged, cost, accuracy, final], feed_dict={x_: np_data, y_: np_targets})
			epoch += 1
			writer.add_summary(_m, epoch)

			if best_cost is None or best_cost > c:
				best_cost = c
				save_path = saver.save(sess, os.path.join(tensorboarddir, "bestsofar.ckpt"))
				print("%d: New best: %s: %.7f (%.7f)" % (epoch, save_path, c, a))

			if (epoch % 128) == 0:
				print(epoch, c, a)

		o, c = sess.run([final, cost], feed_dict={ x_: np_data, y_: np_targets })
		writer.close()

if __name__ == "__main__":
	main()
