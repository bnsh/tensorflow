#! /usr/bin/python

import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def convolution(inp, 

def main():
	with tf.Session() as sess:
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		Xtrain = mnist.train.images
		Ytrain = mnist.train.labels

		Xvalidation = mnist.train.images
		Yvalidation = mnist.train.labels
		
		Xtest = mnist.train.images
		Ytest = mnist.train.labels

		input_feature_sz = Xtrain.shape[1]
		output_feature_sz = Ytrain.shape[1]

		X_ = tf.placeholder(tf.float32, shape=[None, input_feature_sz], name="X")
		Y_ = tf.placeholder(tf.float32, shape=[None, output_feature_sz], name="Y")

		reshaped = tf.reshape(X_, [-1, 28, 28, 1])
		conv1 = convolution(reshaped, [28,28,1],[5,5,32], 0.75)
		conv2 = convolution(conv1, [14,14,32],[5,5,64], 0.75)
		conv3 = convolution(conv2, [7,7,64],[7,7,1024], 0.75)


if __name__ == "__main__":
	main()
