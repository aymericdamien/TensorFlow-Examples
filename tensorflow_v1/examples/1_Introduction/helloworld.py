'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
#THIS IS TO TEST CHANGES AND PUSH GIT
#some mroe code to test discord.....`123`123`123
from __future__ import print_function

import tensorflow as tf

# Simple hello world using TensorFlowjhb

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))
