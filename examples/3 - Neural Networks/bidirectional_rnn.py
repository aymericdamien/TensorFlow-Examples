'''
A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

'''
To classify images using a bidirectional reccurent neural network, we consider every image row as a sequence of pixels.
Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate_fw = tf.placeholder("float", [None, 2*n_hidden])
istate_bw = tf.placeholder("float", [None, 2*n_hidden])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2*n_hidden])),
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([2*n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases, _batch_size, _seq_len):

    # BiRNN requires to supply sequence_length as [batch_size, int64]
    # Note: Tensorflow 0.6.0 requires BiRNN sequence_length parameter to be set
    # For a better implementation with latest version of tensorflow, check below
    _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.int64))

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                            initial_state_fw=_istate_fw,
                                            initial_state_bw=_istate_bw,
                                            sequence_length=_seq_len)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = BiRNN(x, istate_fw, istate_bw, weights, biases, batch_size, n_steps)


# NOTE: The following code is working with current master version of tensorflow
#       BiRNN sequence_length parameter isn't required, so we don't define it
#
# def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases):
#
#     # input shape: (batch_size, n_steps, n_input)
#     _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
#     # Reshape to prepare input to hidden activation
#     _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
#     # Linear activation
#     _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
#
#     # Define lstm cells with tensorflow
#     # Forward direction cell
#     lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
#     # Backward direction cell
#     lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
#     # Split data because rnn cell needs a list of inputs for the RNN inner loop
#     _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
#
#     # Get lstm cell output
#     outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
#                                             initial_state_fw=_istate_fw,
#                                             initial_state_bw=_istate_bw)
#
#     # Linear activation
#     # Get inner loop last output
#     return tf.matmul(outputs[-1], _weights['out']) + _biases['out']
#
# pred = BiRNN(x, istate_fw, istate_bw, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                       istate_bw: np.zeros((batch_size, 2*n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                                istate_bw: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                             istate_bw: np.zeros((batch_size, 2*n_hidden))})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate_fw: np.zeros((test_len, 2*n_hidden)),
                                                             istate_bw: np.zeros((test_len, 2*n_hidden))})
