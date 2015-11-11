import tensorflow as tf

#Simple hello world using TensorFlow

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print sess.run(hello)