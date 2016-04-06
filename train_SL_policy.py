import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

n_input = 361 # go board data input (img shape: 19*19)
n_classes = 361 # go board total classes
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def layer(x, dropout):
    w = tf.Variable(tf.random_normal([5, 5, 1, 128]))
    b = tf.Variable(tf.random_normal([128]))
    conv = conv2d('conv1', x, w, b)
    pool = max_pool('pool1', conv, k=2)
    layer_out = norm('norm1', pool, lsize=4)
    layer_out = tf.nn.dropout(layer_out, dropout)
    return layer_out

def sl_policy_net(_X, dropout):
    _X = tf.reshape(_X, shape=[-1, 19, 19, 1])

    out1 = layer(_X, dropout)
    out2 = layer(out1, dropout)
    out3 = layer(out2, dropout)

    w_d1 = tf.Variable(tf.random_normal([19*19*128, 1000]))
    b_d1 = tf.Variable(tf.random_normal([1000]))
    dense1 = tf.reshape(out3, [-1, w_d1.get_shape().as_list()[0]]) # Reshape out3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, w_d1) + b_d1, name='fc1') # Relu activation

    w_d2 = tf.Variable(tf.random_normal([1000, 500]))
    b_d2 = tf.Variable(tf.random_normal([]))
    dense2 = tf.nn.relu(tf.matmul(dense1, w_d2) + b_d2, name='fc2') # Relu activation

    w_out = tf.Variable(tf.random_normal([500, n_classes]))
    b_out = tf.Variable(tf.random_normal([n_classes]))
    out = tf.matmul(dense2, w_out) + b_out
    return out

pred = sl_policy_net(x, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = kgs.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: kgs.test.images[:256], y: kgs.test.labels[:256], keep_prob: 1.})