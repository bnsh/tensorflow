import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def main():
    tf.set_random_seed(12345)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    Xtrain = mnist.train.images
    Ytrain = mnist.train.labels

    Xvalidation = mnist.validation.images
    Yvalidation = mnist.validation.labels

    Xtest = mnist.test.images
    Ytest = mnist.test.labels

    x_ = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    size1 = 128
    #layer1 = tf.Variable(tf.truncated_normal([784, size1]), name="layer1")
    layer1 = tf.get_variable('layer1', shape=[784, size1])
    bias1 = tf.Variable(tf.random_uniform([size1], 0.1, 0.2, dtype=tf.float32), name="bias1")
    output1 = tf.nn.tanh(tf.matmul(x_, layer1) + bias1)

    size2 = 128
    #layer2 = tf.Variable(tf.truncated_normal([size1, size2]), name="layer2")
    layer2 = tf.get_variable('layer2', shape=[size1, size2])
    bias2 = tf.Variable(tf.random_uniform([size2], 0.1, 0.2, dtype=tf.float32), name="bias2")
    output2 = tf.nn.tanh(tf.matmul(output1, layer2) + bias2)

    size3 = 2
    #layer3 = tf.Variable(tf.truncated_normal([size2, size3]), name="layer3")
    layer3 = tf.get_variable('layer3', shape=[size2, size3])
    bias3 = tf.Variable(tf.random_uniform([size3], 0.1, 0.2, dtype=tf.float32), name="bias3")
    output3 = tf.nn.tanh(tf.matmul(output2, layer3) + bias3)

    size4 = 128
    #layer4 = tf.Variable(tf.truncated_normal([size3, size4]), name="layer4")
    layer4 = tf.get_variable('layer4', shape=[size3, size4])
    bias4 = tf.Variable(tf.random_uniform([size4], 0.1, 0.2, dtype=tf.float32), name="bias4")
    output4 = tf.nn.tanh(tf.matmul(output3, layer4) + bias4)

    size5 = 128
    #layer5 = tf.Variable(tf.truncated_normal([size4, size5]), name="layer5")
    layer5 = tf.get_variable('layer5', shape=[size4, size5])
    bias5 = tf.Variable(tf.random_uniform([size5], 0.1, 0.2, dtype=tf.float32), name="bias5")
    output5 = tf.nn.tanh(tf.matmul(output4, layer5) + bias5)

    size6 = 10
    #layer6 = tf.Variable(tf.truncated_normal([size5, size6]), name="layer6")
    layer6 = tf.get_variable('layer6', shape=[size5, size6])
    bias6 = tf.Variable(tf.random_uniform([size6], 0.1, 0.2, dtype=tf.float32), name="bias6")
    logits6 = tf.matmul(output5, layer6) + bias6
    output6 = tf.nn.softmax(logits6)

    # cost = tf.reduce_mean(tf.pow(y_ - output6, 2) / 2)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits6, labels=y_))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    corrects = tf.equal(tf.argmax(output6, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    writer = tf.summary.FileWriter("/tmp/graphs", sess.graph)

    """
    nocoder_saver = tf.train.Saver({
      "layer1": layer1,
      "layer2": layer2,
      "layer5": layer5,
      "layer6": layer6
    })
    encoder_saver = tf.train.Saver({
      "layer3": layer3,
      "layer4": layer4,
    })

    try:
        #nocoder_saver.restore(sess, "tmp/encoder/checkpoints/nocoder.ckpt")
        print("loaded nocoder")
    except Exception as e:
        print(e)
    try:
        #encoder_saver.restore(sess, "tmp/encoder/checkpoints/encoder.ckpt")
        print("loaded encoder")
    except Exception as e:
        print(e)
    """

    for i in range(1000):
        for mb in xrange(0, 55):
            bx, by = mnist.train.next_batch(1000)
            sess.run([train_step], feed_dict={x_: bx, y_: by})
        outputs, err, cor, acc = sess.run([output6, cost, corrects, accuracy], feed_dict={x_: Xtest, y_: Ytest})
        print("Epoch: %d" % (i,))
        print("\tError: %.7f" % (err,))
        print("\tAccuracy: %.7f" % (acc,))
        # nocode_save_path = nocoder_saver.save(sess, "tmp/unlocked/checkpoints/nocoder.ckpt")
        # encode_save_path = encoder_saver.save(sess, "tmp/unlocked/checkpoints/encoder.ckpt")

    writer.close()

main()   
