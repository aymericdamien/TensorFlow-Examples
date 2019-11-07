## TensorFlow 2.0 Examples

*** More examples to be added later... ***

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

#### 5 - Data Management
- **Load and Parse data** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb)). Build efficient data pipeline with TensorFlow 2.0 (Numpy arrays, Images, CSV files, custom data, ...).
- **Build and Load TFRecords** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb)). Convert data into TFRecords format, and load them with TensorFlow 2.0.
- **Image Transformation (i.e. Image Augmentation)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/image_transformation.ipynb)). Apply various image augmentation techniques with TensorFlow 2.0, to generate distorted images for training.

## Installation

To install TensorFlow 2.0, simply run:
```
pip install tensorflow==2.0.0
```

or (if you want GPU support):
```
pip install tensorflow_gpu==2.0.0
```
