import struct

import tensorflow as tf
import numpy as np


MAGIC_NUMBER_IMAGES = 2051
MAGIC_NUMBER_LABELS = 2049
TRAINING_SIZE = 60000
TEST_SIZE = 10000
NUM_ROWS = 28
NUM_COLUMNS = 28


class MNISTReader(object):
    '''Load/stream mnist training/label data files from yann lecunn's site.'''

    def __init__(self, images_path, labels_path, size=TRAINING_SIZE):
        # Magic number, size as per http://yann.lecun.com/exdb/mnist/
        self.images_fp = self._open_and_validate(images_path, MAGIC_NUMBER_IMAGES, size)
        self.labels_fp = self._open_and_validate(labels_path, MAGIC_NUMBER_LABELS, size)

    def _open_and_validate(self, path, magic, size, rows=NUM_ROWS, columns=NUM_COLUMNS):
        '''Open file and validate magic number/size and return fp.'''

        fp = open(path, 'rb')
        # Magic number and file size are both 32 bit integers
        # 8 bytes total so we read the first 8 bytes of the file
        magic_actual, size_actual = struct.unpack('>ii', fp.read(8))  # Big endian encoding
        if not (
           magic == magic_actual
           and size == size_actual
       ):
            raise Exception("File format not as expected, you sure you're loading the right thing man?")

        # Read and validate image sizes
        if magic == MAGIC_NUMBER_IMAGES:
            rows_actual, columns_actual = struct.unpack('>ii', fp.read(8))
            if not (
               rows == rows_actual
               and rows == columns_actual
           ):
                raise Exception("File format not as expected, you sure you're loading the right thing man?")

        return fp

    def _create_one_hot(self, index):
        one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        one_hot[index - 1] = 1
        return one_hot

    def read(self, number):
        '''Read `number` of entries from the data.'''
        images = []
        labels = []
        for i in xrange(0, number):
            # Get an image array from `images_fp`
            image_array = []
            # Pixels are organized on a per row basis
            for j in xrange(0, NUM_ROWS):
                row = list(struct.unpack('>' + 'B' * NUM_COLUMNS, self.images_fp.read(NUM_COLUMNS)))
                image_array += row
            images.append(image_array)

            # Get a label from `labels_fp`
            label = struct.unpack('>B', self.labels_fp.read(1))[0]
            # One hot encoding of the label
            labels.append(self._create_one_hot(label))
        return images, labels

# rescale from whatever it was to 0..1
def rescale(x):
    x = np.array(x, dtype=np.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn)

def myinitializer(shape):
#    return tf.truncated_normal(shape)
    return (tf.random_uniform(shape) * 2 - 1) * np.sqrt(6.0 / (shape[0] + shape[1]))

with tf.name_scope('mnist'):
    tf.set_random_seed(12345)

    x_ = tf.placeholder(tf.float32, [None, 784], name="image")
    y_ = tf.placeholder(tf.float32, [None, 10], name="label")

    w1 = tf.Variable(myinitializer([784, 100]), name="weight1")
    b1 = tf.Variable(tf.zeros([1, 100]), name="bias1")

    y1 = tf.nn.tanh(tf.matmul(x_, w1) + b1)

    w2 = tf.Variable(myinitializer([100, 2]), name="weight2")
    b2 = tf.Variable(tf.zeros([1, 2]), name="bias2")

    y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)

    w3 = tf.Variable(myinitializer([2, 10]), name="weight3")
    b3 = tf.Variable(tf.zeros([1, 10]), name="bias3")

    y3 = tf.matmul(y2, w3) + b3

    #cost = tf.reduce_mean(tf.pow(y_ - y1, 2) / 2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y3, labels=y_))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y3,1), tf.argmax(y_,1)), dtype=tf.float32))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Run through the data set 10 times
    for rnd in range(0, 100):
        mnist = MNISTReader('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
        for i in range(60):
            x_in, y_out = mnist.read(1000)
            x_in = rescale(x_in)
            #out, err, _ = sess.run([y1, cost, train_step], feed_dict={x_: x_in, y_: y_out})
            err, a, _ = sess.run([cost, accuracy, train_step], feed_dict={x_: x_in, y_: y_out})
            if i % 1000 == 0:
                print """
   Error: {}
Accuracy: {}
                """.format(err, a)

