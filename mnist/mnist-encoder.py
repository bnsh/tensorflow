#! /usr/bin/python

import sys
import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def dump_json(epoch, encodings, targets, outputs):
	arr = []
	for i in xrange(0, targets.shape[0]):
		p = [int(targets[i]), float(encodings[i, 0]), float(encodings[i, 1]), int(outputs[i])]
		arr.append(p)
	with open("/tmp/processing/%08d.json" % (epoch), "w") as fp:
		json.dump(arr, fp)


def create_layer(inputsz, outputsz, activation, keep, name, X, wname=None, reuse=None):
	rv = { }
	if wname is None:
		wname = name

	print wname, name

	with tf.variable_scope(name) as scope:
# Lesson learned: DON'T Initialize this way!!
#		W = tf.get_variable("W", [inputsz, outputsz], dtype=tf.float32, initializer=tf.random_normal_initializer(dtype=tf.float32))
		W = tf.get_variable("W", [inputsz, outputsz], dtype=tf.float32)
		# b = tf.get_variable("b", [1, outputsz], dtype=tf.float32, initializer=tf.random_uniform_initializer(0.0,0.0,dtype=tf.float32))
		b = tf.get_variable("b", [1, outputsz], dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.float32))
		z = tf.matmul(X, W) + b
		if activation is None:
			activation = lambda x: x
		if keep is not None:
			y = tf.nn.dropout(activation(z), keep_prob=keep)
		else:
			y = activation(z)
		if False and (inputsz == outputsz):
			y = y + X
	return y
			
	

def main():
	# Just make some directories...
	if not os.path.exists("/tmp/processing"):
		os.mkdir("/tmp/processing", 0775)
	if not os.path.exists("/tmp/graphs"):
		os.mkdir("/tmp/graphs", 0775)
	with tf.Session() as sess:
		tf.set_random_seed(12345)
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
		keep_prob_ = tf.placeholder(tf.float32, name="keep_prob")

		hiddensz = 100
		encodeszs = [2]
		layers = 0
		keep = 1.0
		activation = tf.nn.tanh
		lr = 0.01

		output = create_layer(input_feature_sz, hiddensz, activation, keep_prob_, "input_layer", X_)

		for i in xrange(0, layers):
			output = create_layer(hiddensz, hiddensz, activation, keep_prob_, "mid-pre-%d" % (1+i), output)

		output_cpy = output
		encodings = { }
		errors = { }
		accuracies = { }
		classifications = { }
		for j, encodesz in enumerate(encodeszs):
			output = output_cpy
			output = create_layer(hiddensz, encodesz, tf.nn.tanh, None, "encoder-%d" % (encodesz), output)
			encodings[encodesz] = output
			output = create_layer(encodesz, hiddensz, activation, keep_prob_, "decoder-%d" % (encodesz), output)

			for i in xrange(0, layers):
				title="mid-post-%d" % (1+i)
				output = create_layer(hiddensz, hiddensz, activation, keep_prob_, "%s-%d" % (title, encodesz), output, title, (j != 0))

			output = create_layer(hiddensz, output_feature_sz, None, None, "output-%d" % (encodesz), output, "output", (j != 0))

			with tf.variable_scope("error-%d" % (encodesz)):
				errors[encodesz] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y_))
				classifications[encodesz] = tf.argmax(output, 1)
				corrects = tf.equal(classifications[encodesz], tf.argmax(Y_, 1))
				accuracies[encodesz] = tf.reduce_mean(tf.cast(corrects, tf.float32))

		with tf.variable_scope("errors"):
			# total_loss = tf.reduce_sum(tf.pow(encodings[2], 2))
			total_loss = None
			for i, esz in enumerate(errors):
				if total_loss == None:
					total_loss = errors[esz]
				else:
					total_loss += errors[esz]
			opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)


		tf.summary.scalar("totalloss", total_loss)
		for sz in encodeszs:
			tf.summary.scalar("accuracy%d" % (sz), accuracies[sz])
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("/tmp/graphs/%s" % (time.strftime("%Y%m%dT%H%M%S", time.localtime(time.time()))), sess.graph)
		sess.run(tf.global_variables_initializer())
		done = False
		epoch = 0
		tl, a, e, c, m = sess.run([total_loss, accuracies, encodings, classifications, merged], feed_dict={X_: Xtest, Y_: Ytest, keep_prob_: 1.0})
		sys.stderr.write("Epoch: %d: tl=%.7f" % (epoch, tl))
		for sz in sorted(a.keys()):
			sys.stderr.write("	a%d=%.7f" % (sz, a[sz]))
		sys.stderr.write("\n")
		while not done:
			epoch += 1
			for mb in xrange(0, 55):
				bx, by = mnist.train.next_batch(1000)
				sess.run(opt, feed_dict={X_: bx, Y_: by, keep_prob_: keep})
			tl, a, e, c, m = sess.run([total_loss, accuracies, encodings, classifications, merged], feed_dict={X_: Xtest, Y_: Ytest, keep_prob_: 1.0})
			dump_json(epoch, e[2], np.argmax(Ytest, 1), c[2])
			writer.add_summary(m, epoch)
			sys.stderr.write("Epoch: %d: tl=%.7f" % (epoch, tl))
			for sz in sorted(a.keys()):
				sys.stderr.write("	a%d=%.7f" % (sz, a[sz]))
			sys.stderr.write("\n")
		writer.close()

if __name__ == "__main__":
	main()
