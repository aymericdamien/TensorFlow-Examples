# TensorFlow Examples

This tutorial was designed for easily diving into TensorFlow, through examples. For readability, it includes both notebooks and source codes with explanation, for both TF v1 & v2.

It is suitable for beginners who want to find clear and concise examples about TensorFlow. Besides the traditional 'raw' TensorFlow implementations, you can also find the latest TensorFlow API practices (such as `layers`, `estimator`, `dataset`, ...).

**Update (05/16/2020):** Moving all default examples to TF2. For TF v1 examples: [check here](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1).

## Tutorial index

#### 0 - Prerequisite
- [Introduction to Machine Learning](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/0_Prerequisite/ml_introduction.ipynb).
- [Introduction to MNIST Dataset](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

#### 1 - Introduction
- **Hello World** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb)). Very simple example to learn how to print "hello world" using TensorFlow 2.0.
- **Basic Operations** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb)). A simple example that cover TensorFlow 2.0 basic operations.

#### 2 - Basic Models
- **Linear Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb)). Implement a Linear Regression with TensorFlow 2.0.
- **Logistic Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb)). Implement a Logistic Regression with TensorFlow 2.0.
- **Word2Vec (Word Embedding)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/word2vec.ipynb)). Build a Word Embedding Model (Word2Vec) from Wikipedia data, with TensorFlow 2.0.
- **GBDT (Gradient Boosted Decision Trees)** ([notebooks](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/gradient_boosted_trees.ipynb)). Implement a Gradient Boosted Decision Trees with TensorFlow 2.0+ to predict house value using Boston Housing dataset.

#### 3 - Neural Networks
##### Supervised

- **Simple Neural Network** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb)). Use TensorFlow 2.0 'layers' and 'model' API to build a simple neural network to classify MNIST digits dataset.
- **Simple Neural Network (low-level)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb)). Raw implementation of a simple neural network to classify MNIST digits dataset.
- **Convolutional Neural Network** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb)). Use TensorFlow 2.0 'layers' and 'model' API to build a convolutional neural network to classify MNIST digits dataset.
- **Convolutional Neural Network (low-level)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)). Raw implementation of a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/recurrent_network.ipynb)). Build a recurrent neural network (LSTM) to classify MNIST digits dataset, using TensorFlow 2.0 'layers' and 'model' API.
- **Bi-directional Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)). Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset, using TensorFlow 2.0 'layers' and 'model' API.
- **Dynamic Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)). Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of variable length, using TensorFlow 2.0 'layers' and 'model' API.

##### Unsupervised
- **Auto-Encoder** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - Utilities
- **Save and Restore a model** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/save_restore_model.ipynb)). Save and Restore a model with TensorFlow 2.0.
- **Build Custom Layers & Modules** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/build_custom_layers.ipynb)). Learn how to build your own layers / modules and integrate them into TensorFlow 2.0 Models.
- **Tensorboard** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/tensorboard.ipynb)). Track and visualize neural network computation graph, metrics, weights and more using TensorFlow 2.0+ tensorboard.

#### 5 - Data Management
- **Load and Parse data** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb)). Build efficient data pipeline with TensorFlow 2.0 (Numpy arrays, Images, CSV files, custom data, ...).
- **Build and Load TFRecords** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb)). Convert data into TFRecords format, and load them with TensorFlow 2.0.
- **Image Transformation (i.e. Image Augmentation)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/image_transformation.ipynb)). Apply various image augmentation techniques with TensorFlow 2.0, to generate distorted images for training.


## TensorFlow v1

The tutorial index for TF v1 is available here: [TensorFlow v1.15 Examples](tensorflow_v1). Or see below for a list of the examples.

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples.
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [this notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

Official Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

## Installation

To download all the examples, simply clone this repository:
```
git clone https://github.com/aymericdamien/TensorFlow-Examples
```

To run them, you also need the latest version of TensorFlow. To install it:
```
pip install tensorflow
```

or (with GPU support):
```
pip install tensorflow_gpu
```

For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://www.tensorflow.org/install/)


## TensorFlow v1 Examples - Index

The tutorial index for TF v1 is available here: [TensorFlow v1.15 Examples](tensorflow_v1).

#### 0 - Prerequisite
- [Introduction to Machine Learning](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/tensorflow_v1/0_Prerequisite/ml_introduction.ipynb).
- [Introduction to MNIST Dataset](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/tensorflow_v1/0_Prerequisite/mnist_dataset_intro.ipynb).

#### 1 - Introduction
- **Hello World** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/1_Introduction/helloworld.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/1_Introduction/helloworld.py)). Very simple example to learn how to print "hello world" using TensorFlow.
- **Basic Operations** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/tensorflow_v1/1_Introduction/basic_operations.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-examples/Examples/blob/master/tensorflow_v1/1_Introduction/basic_operations.py)). A simple example that cover TensorFlow basic operations.
- **TensorFlow Eager API basics** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/1_Introduction/basic_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/1_Introduction/basic_eager_api.py)). Get started with TensorFlow's Eager API.

#### 2 - Basic Models
- **Linear Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/linear_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/linear_regression.py)). Implement a Linear Regression with TensorFlow.
- **Linear Regression (eager api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/linear_regression_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/linear_regression_eager_api.py)). Implement a Linear Regression using TensorFlow's Eager API.
- **Logistic Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/logistic_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/logistic_regression.py)). Implement a Logistic Regression with TensorFlow.
- **Logistic Regression (eager api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/logistic_regression_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/logistic_regression_eager_api.py)). Implement a Logistic Regression using TensorFlow's Eager API.
- **Nearest Neighbor** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/nearest_neighbor.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/nearest_neighbor.py)). Implement Nearest Neighbor algorithm with TensorFlow.
- **K-Means** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/kmeans.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/kmeans.py)). Build a K-Means classifier with TensorFlow.
- **Random Forest** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/random_forest.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/random_forest.py)). Build a Random Forest classifier with TensorFlow.
- **Gradient Boosted Decision Tree (GBDT)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/gradient_boosted_decision_tree.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/gradient_boosted_decision_tree.py)). Build a Gradient Boosted Decision Tree (GBDT) with TensorFlow.
- **Word2Vec (Word Embedding)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/word2vec.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/2_BasicModels/word2vec.py)). Build a Word Embedding Model (Word2Vec) from Wikipedia data, with TensorFlow.

#### 3 - Neural Networks
##### Supervised

- **Simple Neural Network** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/3_NeuralNetworks/notebooks/neural_network_raw.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/neural_network_raw.py)). Build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Simple Neural Network (tf.layers/estimator api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/neural_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/neural_network.py)). Use TensorFlow 'layers' and 'estimator' API to build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset.
- **Simple Neural Network (eager api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/neural_network_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/neural_network_eager_api.py)). Use TensorFlow Eager API to build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset.
- **Convolutional Neural Network** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/convolutional_network_raw.py)). Build a convolutional neural network to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Convolutional Neural Network (tf.layers/estimator api)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/convolutional_network.py)). Use TensorFlow 'layers' and 'estimator' API to build a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/recurrent_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/recurrent_network.py)). Build a recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Bi-directional Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/bidirectional_rnn.py)). Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Dynamic Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/dynamic_rnn.py)). Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of different length.

##### Unsupervised
- **Auto-Encoder** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/autoencoder.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/autoencoder.py)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **Variational Auto-Encoder** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/variational_autoencoder.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/variational_autoencoder.py)). Build a variational auto-encoder (VAE), to encode and generate images from noise.
- **GAN (Generative Adversarial Networks)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/gan.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/gan.py)). Build a Generative Adversarial Network (GAN) to generate images from noise.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/3_NeuralNetworks/dcgan.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/dcgan.py)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - Utilities
- **Save and Restore a model** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/4_Utils/save_restore_model.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/4_Utils/save_restore_model.py)). Save and Restore a model with TensorFlow.
- **Tensorboard - Graph and loss visualization** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/4_Utils/tensorboard_basic.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/4_Utils/tensorboard_basic.py)). Use Tensorboard to visualize the computation Graph and plot the loss.
- **Tensorboard - Advanced visualization** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/4_Utils/tensorboard_advanced.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/4_Utils/tensorboard_advanced.py)). Going deeper into Tensorboard; visualize the variables, gradients, and more...

#### 5 - Data Management
- **Build an image dataset** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/5_DataManagement/build_an_image_dataset.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/5_DataManagement/build_an_image_dataset.py)). Build your own images dataset with TensorFlow data queues, from image folders or a dataset file.
- **TensorFlow Dataset API** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/5_DataManagement/tensorflow_dataset_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/5_DataManagement/tensorflow_dataset_api.py)). Introducing TensorFlow Dataset API for optimizing the input data pipeline.
- **Load and Parse data** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/5_DataManagement/load_data.ipynb)). Build efficient data pipeline (Numpy arrays, Images, CSV files, custom data, ...).
- **Build and Load TFRecords** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/5_DataManagement/tfrecords.ipynb)). Convert data into TFRecords format, and load them.
- **Image Transformation (i.e. Image Augmentation)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/5_DataManagement/image_transformation.ipynb)). Apply various image augmentation techniques, to generate distorted images for training.

#### 6 - Multi GPU
- **Basic Operations on multi-GPU** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/6_MultiGPU/multigpu_basics.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/6_MultiGPU/multigpu_basics.py)). A simple example to introduce multi-GPU in TensorFlow.
- **Train a Neural Network on multi-GPU** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/6_MultiGPU/multigpu_cnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/6_MultiGPU/multigpu_cnn.py)). A clear and simple TensorFlow implementation to train a convolutional neural network on multiple GPUs.
