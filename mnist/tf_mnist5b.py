
import tensorflow as tf
import time
logdir = "graphs/" + time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())) + "/"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./saved_model/model.ckpt.meta')
saver.restore(sess, './saved_model/model.ckpt')

x, y, keep, a, c = tf.get_collection("vars")

classifications, accuracy = sess.run([c, a], feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep: 1})
print("Validation accuracy: %.7f" % (accuracy))

classifications, accuracy = sess.run([c, a], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep: 1})
print("Test accuracy: %.7f" % (accuracy))
