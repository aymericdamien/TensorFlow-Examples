'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import numpy as np
import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(500)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# Reshape images to 1D
Xtr = np.reshape(Xtr, newshape=(-1, 28*28))
Ytr = np.reshape(Ytr, newshape=(-1,))
Xte = np.reshape(Xte, newshape=(-1, 28*28))
Yte = np.reshape(Yte, newshape=(-1,))

xtr = tf.placeholder(tf.float32, [None, 784], name='xtr')
ytr = tf.placeholder(tf.int32, [None,1], name='ytr')
xte = tf.placeholder(tf.float32, [784], name='xte')
yte = tf.placeholder(tf.int32, name='yte')

distance = tf.reduce_sum(tf.abs(tf.sub(xtr, xte)), reduction_indices=1)
value, indices = tf.nn.top_k(distance, 60)
result = tf.nn.embedding_lookup(ytr, indices)
y, _, count = tf.unique_with_counts(tf.reshape(result, (-1,)))
idx = tf.argmax(count, 0)
pred = tf.equal(tf.nn.embedding_lookup(y,idx), yte)

accuracy = 0.
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for item, result in zip(Xte,Yte):
        numpy_pred = sess.run(pred, feed_dict={xtr:Xtr, ytr:Ytr.reshape((-1,1)), xte:item, yte:result})
        if numpy_pred:
            accuracy += 1./len(Xte)
    print "Done!"
    print "Accuracy", accuracy

