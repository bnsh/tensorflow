
import tensorflow as tf
import time
logdir = "graphs/" + time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())) + "/"

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(12345)

mnist = input_data.read_data_sets("C:\\Users\\kubiro1\\Documents\\IPython Notebooks\\Deep Learning\\MNIST_data", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(shape=[None, 784], dtype = tf.float32, name = 'x')
y = tf.placeholder(shape=[None, 10], dtype = tf.float32, name = 'y')
keep = tf.placeholder(tf.float32, name = "keep")



with tf.variable_scope("conv1") as scope:
    reshape = tf.reshape(x, [-1,28,28,1])
    w1 = tf.get_variable("w1", [5,5,1,32], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("b1", [32], initializer = tf.random_normal_initializer())
    conv1 = tf.nn.conv2d(reshape,w1,strides=[1,1,1,1],padding = "SAME")
    conv1_actv = tf.nn.relu(conv1 + b1, name = scope.name)
    print(w1)
with tf.variable_scope("pool1") as scope:
    maxpool1 = tf.nn.max_pool(conv1_actv,ksize = [1,2,2,1], strides = [1,2,2,1],padding = "SAME")
with tf.variable_scope("conv2") as scope:
    w2 = tf.get_variable("w2", [5,5,32,64], initializer=tf.truncated_normal_initializer())
    b2 = tf.get_variable("b2", [64], initializer = tf.random_normal_initializer())
    conv2 = tf.nn.conv2d(maxpool1,w2,strides=[1,1,1,1],padding = "SAME")
    conv2_actv = tf.nn.relu(conv2 + b2)
with tf.variable_scope("pool2") as scope:
   maxpool2 = tf.nn.max_pool(conv2_actv,ksize = [1,2,2,1], strides = [1,2,2,1],padding = "SAME")

with tf.variable_scope("fc") as scope:
    w3 = tf.get_variable("w3", [7*7*64,1024], initializer=tf.truncated_normal_initializer())
    b3 = tf.get_variable("b3", [1024], initializer = tf.random_normal_initializer())
    w4 = tf.get_variable("w4", [1024,10], initializer=tf.truncated_normal_initializer())
    b4 = tf.get_variable("b4", [10], initializer = tf.random_normal_initializer())
    maxpool2_flat = tf.reshape(maxpool2,[-1,7*7*64])
    maxpool2_flat_actv = tf.nn.relu(tf.matmul(maxpool2_flat,w3) + b3)
    maxpool2_dropout = tf.nn.dropout(maxpool2_flat_actv, keep) ##??
    z = tf.add(tf.matmul(maxpool2_dropout, w4), b4, name="z") ##logit

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z, y))
    tf.summary.scalar("loss",loss)
    tf.summary.histogram("w1", w1)
    tf.summary.histogram("b1", b1)
opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

with tf.name_scope("accuracy"):
    classifications = tf.argmax(z, 1, name="classifications")
    correct_prediction = tf.equal(classifications, tf.argmax(y,1))  #??
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

sess.run(tf.global_variables_initializer())
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logdir,  sess.graph)

for i in (x, y, keep, accuracy, classifications):
    tf.add_to_collection("vars", i)

saver = tf.train.Saver()
best_acc = None
for i in range(2):
  for j in range(55):
    batch_x, batch_y = mnist.train.next_batch(100)
    l, _ = sess.run([loss,opt], feed_dict= {x : batch_x, y : batch_y, keep: 0.75}) ##?
    summary_str = sess.run(merged_summary_op, feed_dict= {x : batch_x, y : batch_y, keep: 0.75})
    summary_writer.add_summary(summary_str, i * 55 + j )
#  if i % 1000 == 0: 
  l, acc = sess.run([loss, accuracy], feed_dict={x: mnist.validation.images, y:mnist.validation.labels, keep: 1})
  print (i,":",l,":",acc)
  if best_acc is None or acc > best_acc:
    best_acc = acc
    save_path = saver.save(sess, './saved_model/model.ckpt')
    print("model saved in : %s" % save_path)

summary_writer.close()

