# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#Network Parameters
n_input = 784 #MNIST data input
n_classes = 10 #MNIST total classes
dropout = 0.75

# Create model
x = tf.placeholder(tf.types.float32, [None, n_input])
y = tf.placeholder(tf.types.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.types.float32) #dropout

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    conv1 = max_pool(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, _dropout)

    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pool(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, _dropout)

    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'])
    dense1 = tf.nn.dropout(dense1, _dropout)

    out = tf.matmul(dense1, _weights['out']) + _biases['out']
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))

# Train
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})/batch_size
            print "Iter", str(step*batch_size), "loss=", "{:.9f}".format(loss/step)
        step += 1
    print "Optimization Finished!"
    #Accuracy on 256 mnist test images
    print "Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
