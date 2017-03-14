import tensorflow as tf

with tf.name_scope('xor'):
    x_ = tf.placeholder(tf.float32, [2, 2])
    y_ = tf.placeholder(tf.float32, [2, 1])
    w1 = tf.Variable(tf.truncated_normal([2, 2]))
    b1 = tf.Variable(tf.zeros([2]))

    y1 = tf.nn.sigmoid(tf.matmul(x_, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([2, 1]))
    b2 = tf.Variable(tf.zeros([1]))

    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
    cost = tf.reduce_mean(tf.pow(y2 - y_, 2) /2)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    x_in = [
        [0, 1],
        [1, 1],
    ]
    x_in_2 = [
        [0, 0],
        [1, 0],
    ]
    y_out = [
        [1],
        [0],
    ]
    y_out_2 = [
        [0],
        [1],
    ]

    for i in range(10000):
        if i % 2 == 0:
            x = x_in
            y = y_out
        else:
            x = x_in_2
            y = y_out_2

        sess.run(train_step, feed_dict={x_: x, y_: y})
	
	# Tried checking the results after training via
	# sess.run(y2, feed_dict={x_: x_in})
	# sess.run(y2, feed_dict={x_: x_in_2})
