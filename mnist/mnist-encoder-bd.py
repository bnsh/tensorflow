import struct

# As far as I can tell, the only difference between
# Bo's and mine was that mine used tensorflow's mnist data
# which has values from 0..1 and he used his own, which has values
# from 0..255 and B. I used a better initializer for the weights.
# But, the better initializer isn't that much more of an advantage..
# It speeds up the training time to be sure, but without it,
# Bo's version gets to 88% within 100 epochs.
# With Xavier initialization tho, I get to 93% within 100 epochs.

import json
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def rescale(X):
    mn = X.min()
    mx = X.max()
    X = ((X - mn) / (mx - mn))
    return X

Xtrain = rescale(mnist.train.images)
Ytrain = mnist.train.labels

Xvalidation = rescale(mnist.validation.images)
Yvalidation = mnist.validation.labels


def dump_json(epoch, encodings, targets, outputs):
    arr = []
    for i in xrange(0, targets.shape[0]):
        p = [int(targets[i]), float(encodings[i, 0]), float(encodings[i, 1]), int(outputs[i])]
        arr.append(p)
    with open("/tmp/bo/data/%08d.json" % (epoch), "w") as fp:
        json.dump(arr, fp)


def myxavier(shape):
# https://www.tensorflow.org/versions/master/api_docs/python/contrib.layers/initializers
    rv = (tf.random_uniform(shape=shape) * 2 - 1) * np.sqrt(6.0 / (shape[0] + shape[1]))
    # rv = (tf.truncated_normal(shape=shape))
    return rv

with tf.name_scope('mnist'):
    tf.set_random_seed(12345)

    x_ = tf.placeholder(tf.float32, [None, 784], name="image")
    y_ = tf.placeholder(tf.float32, [None, 10], name="label")

    w1 = tf.Variable(myxavier([784, 100]), name="weight1")
    b1 = tf.Variable(tf.zeros([1, 100]), name="bias1")

    y1 = tf.nn.tanh(tf.matmul(x_, w1) + b1)

    w2 = tf.Variable(myxavier([100, 2]), name="weight2")
    b2 = tf.Variable(tf.zeros([1, 2]), name="bias2")

    y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)

    w3 = tf.Variable(myxavier([2, 10]), name="weight3")
    b3 = tf.Variable(tf.zeros([1, 10]), name="bias3")
    y3 = tf.matmul(y2, w3) + b3

    #cost = tf.reduce_mean(tf.pow(y_ - y1, 2) / 2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y3, labels=y_))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(y3,1)), dtype=tf.float32))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Run through the data set 10 times
    for rnd in range(0, 100):
        for i in range(55):
            lo = i * 1000
            hi = (i+1) * 1000
            x_in = Xtrain[lo:hi,:]
            y_out = Ytrain[lo:hi,:]
            #out, err, _ = sess.run([y1, cost, train_step], feed_dict={x_: x_in, y_: y_out})
            acc, out, err, _ = sess.run([accuracy, y3, cost, train_step], feed_dict={x_: x_in, y_: y_out})

        coordinates, predictions, acc, err = sess.run([y2, y3, accuracy, cost], feed_dict={x_: Xvalidation, y_: Yvalidation})
        print """
     Error: {}
  Accuracy: {}
                """.format(err, acc)
        dump_json(rnd, coordinates, np.argmax(Yvalidation, 1), np.argmax(predictions,1))

