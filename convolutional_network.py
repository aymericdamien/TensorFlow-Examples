# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 3
batch_size = 64
display_batch = 200 #set to 0 to turn off
display_step = 1

#Network Parameters
n_input = 784 #MNIST data input
n_classes = 10 #MNIST total classes

# Create model
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME') + b)

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases):
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    conv1 = max_pool(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, 0.75)

    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pool(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, 0.75)

    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'])
    dense1 = tf.nn.dropout(dense1, 0.75)

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

pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Train
#load mnist data
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    #one epoch can take a long time on CPU
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            if i % display_batch == 0 and display_batch > 0:
                print "Epoch:", '%04d' % (epoch+1), "Batch " + str(i) + "/" + str(total_batch), "cost=", \
                    "{:.9f}".format(sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
