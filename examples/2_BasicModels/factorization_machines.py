""" Factorisation Machines for Multiclass Classification.
A simple factorization machine algorithm implementation with TensorFlow. 
This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
The implementation includes regularisation of the parameters for the weights 
as well as the interaction factors. It also includes eta (learning rate) parameter  

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from sklearn import metrics
import tensorflow as tf

# Parameters
learning_rate = 0.01
num_steps = 8000
batch_size = 256
display_step = 100

# FM Parameters
number_latent = 5
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# Regularization terms for W and V matrices
lambda_w = tf.constant(0.0015, name='lambda_w')
lambda_v = tf.constant(0.0015, name='lambda_v')

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'W': tf.Variable(tf.random_normal([num_input, num_classes])),
    'b': tf.Variable(tf.random_normal([num_classes]))
}
interactions = {
    'V': tf.Variable(tf.random_normal([num_input, number_latent, num_classes])),
}

# Create model
def fm(x):
    # Linear terms 
    linear_terms = tf.add(tf.matmul(x, weights['W']), weights['b'])
    # Simplification follows: Rendle, 2010 and Rendle, 2012.
    
    # Squered sum of product (using tensordot as we are having tensors 
    # instead of matrices for multiclass classification)
    s1 = tf.pow(
        tf.reduce_sum(
            tf.tensordot(
                x, 
                interactions['V'], [[1], [0]]
            ), 1
        ), 2)
    
    # Sum of squared product 
    s2 = tf.reduce_sum(
        tf.tensordot(
            tf.pow(x, 2), 
            tf.pow(interactions['V'], 2), 
            [[1], [0]]
        ), 1)
    outputs = linear_terms + 1/2*(s1-s2)
    return outputs

# Construct model
logits = fm(X)
prediction = tf.nn.softmax(logits)

# Define loss, regularisation, and optimizer
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))


regularisation = tf.add(tf.reduce_sum(lambda_v * tf.math.abs(interactions['V'])),
                        tf.reduce_sum(lambda_w * tf.pow(weights['W'],2)))
loss_op = error + regularisation
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
cm = tf.math.confusion_matrix(tf.argmax(Y, 1), tf.argmax(prediction, 1))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        if step%1000 == 0:
            learning_rate = learning_rate/2
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([error, accuracy], feed_dict={X: batch_x, Y: batch_y})
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    print("Testing Confusion Matrix: \n", \
        sess.run(cm, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
