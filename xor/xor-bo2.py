import tensorflow as tf

with tf.name_scope('xor'):
    x_ = tf.placeholder(tf.float32, [None, 2])
    y_ = tf.placeholder(tf.float32, [None, 1])
    w1 = tf.Variable(tf.truncated_normal([2, 2]))
    b1 = tf.Variable(tf.zeros([2]))

    y1 = tf.nn.sigmoid(tf.matmul(x_, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([2, 1]))
    b2 = tf.Variable(tf.zeros([1]))

    y2 = tf.matmul(y1, w2) + b2
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y2, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    x_in = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y_out = [
        [0],
        [1],
        [1],
        [0]
    ]

    for i in range(10000):
        out, c, _ = sess.run([y2, cost, train_step], feed_dict={x_: x_in, y_: y_out})
	if i % 1024 == 0:
		print "cost: %.7f" % (c)
		print out
	
	# Tried checking the results after training via
	# sess.run(y2, feed_dict={x_: x_in})
	# sess.run(y2, feed_dict={x_: x_in_2})
