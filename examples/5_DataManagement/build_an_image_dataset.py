""" Build an Image Dataset in TensorFlow.

For this example, you need to make your own set of images.
We will show 2 different ways to build that dataset:

- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |             
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```

Below, there are some parameters that you need to change (Marked 'CHANGE HERE'), 
such as the dataset path.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

import tensorflow as tf

# Dataset Parameters
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
DATASET_PATH = '/dataset/path' # CHANGE HERE, the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

# Testing (Optional)
TEST_IMAGE = '' # CHANGE HERE, give a path to an image to test the model


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        # Collect all paths and labels
        imagepaths, labels = list(), list()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        pass
    else:
        raise Exception("Unknown mode.")
    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_image(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(tf.cast(image, tf.float32), IMG_HEIGHT, IMG_WIDTH)

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                  capacity=batch_size * 8,
                                  min_after_dequeue=batch_size * 2)

    return X, Y

# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 10

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
X, Y = read_images(DATASET_PATH, MODE, batch_size)

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)
    step = 1

    # Start the data queue
    tf.train.start_queue_runners()

    # Keep training until reach max iterations
    while step < num_steps:

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)
        step += 1
    print("Optimization Finished!")

    # # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
    #                                   Y: mnist.test.labels[:256]}))
