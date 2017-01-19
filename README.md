# TensorFlow Examples
TensorFlow Tutorial with popular machine learning algorithms implementation. This tutorial was designed for easily diving into TensorFlow, through examples.

It is suitable for beginners who want to find clear and concise examples about TensorFlow. For readability, the tutorial includes both notebook and code with explanations.

Note: If you are using older TensorFlow version (before 0.12), please have a [look here](https://github.com/aymericdamien/TensorFlow-Examples/tree/0.11)

## Tutorial index

#### 0 - Prerequisite
- Introduction to Machine Learning ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/ml_introduction.ipynb))
- Introduction to MNIST Dataset ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb))

#### 1 - Introduction
- Hello World ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/helloworld.py))
- Basic Operations ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/basic_operations.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_operations.py))

#### 2 - Basic Models
- Nearest Neighbor ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py))
- Linear Regression ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py))
- Logistic Regression ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py))

#### 3 - Neural Networks
- Multilayer Perceptron ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py))
- Convolutional Neural Network ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py))
- Recurrent Neural Network (LSTM) ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py))
- Bidirectional Recurrent Neural Network (LSTM) ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py))
- Dynamic Recurrent Neural Network (LSTM) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py))
- AutoEncoder ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py))

#### 4 - Utilities
- Save and Restore a model ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4_Utils/save_restore_model.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py))
- Tensorboard - Graph and loss visualization ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4_Utils/tensorboard_basic.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py))
- Tensorboard - Advanced visualization ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py))

#### 5 - Multi GPU
- Basic Operations on multi-GPU ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5_MultiGPU/multigpu_basics.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_MultiGPU/multigpu_basics.py))

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples (with input_data.py).
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [this notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

Official Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## More Examples
The following examples are coming from [TFLearn](https://github.com/tflearn/tflearn), a library that provides a simplified interface for TensorFlow. You can have a look, there are many [examples](https://github.com/tflearn/tflearn/tree/master/examples) and [pre-built operations and layers](http://tflearn.org/doc_index/#api).

### Tutorials
- [TFLearn Quickstart](https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md). Learn the basics of TFLearn through a concrete machine learning task. Build and train a deep neural network classifier.

### Basics
- [Linear Regression](https://github.com/tflearn/tflearn/blob/master/examples/basics/linear_regression.py). Implement a linear regression using TFLearn.
- [Logical Operators](https://github.com/tflearn/tflearn/blob/master/examples/basics/logical.py). Implement logical operators with TFLearn (also includes a usage of 'merge').
- [Weights Persistence](https://github.com/tflearn/tflearn/blob/master/examples/basics/weights_persistence.py). Save and Restore a model.
- [Fine-Tuning](https://github.com/tflearn/tflearn/blob/master/examples/basics/finetuning.py). Fine-Tune a pre-trained model on a new task.
- [Using HDF5](https://github.com/tflearn/tflearn/blob/master/examples/basics/use_hdf5.py). Use HDF5 to handle large datasets.
- [Using DASK](https://github.com/tflearn/tflearn/blob/master/examples/basics/use_dask.py). Use DASK to handle large datasets.

### Computer Vision
- [Multi-layer perceptron](https://github.com/tflearn/tflearn/blob/master/examples/images/dnn.py). A multi-layer perceptron implementation for MNIST classification task.
- [Convolutional Network (MNIST)](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py). A Convolutional neural network implementation for classifying MNIST dataset.
- [Convolutional Network (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py). A Convolutional neural network implementation for classifying CIFAR-10 dataset.
- [Network in Network](https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py). 'Network in Network' implementation for classifying CIFAR-10 dataset.
- [Alexnet](https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py). Apply Alexnet to Oxford Flowers 17 classification task.
- [VGGNet](https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py). Apply VGG Network to Oxford Flowers 17 classification task.
- [VGGNet Finetuning (Fast Training)](https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network_finetuning.py). Use a pre-trained VGG Network and retrain it on your own data, for fast training.
- [RNN Pixels](https://github.com/tflearn/tflearn/blob/master/examples/images/rnn_pixels.py). Use RNN (over sequence of pixels) to classify images.
- [Highway Network](https://github.com/tflearn/tflearn/blob/master/examples/images/highway_dnn.py). Highway Network implementation for classifying MNIST dataset.
- [Highway Convolutional Network](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_highway_mnist.py). Highway Convolutional Network implementation for classifying MNIST dataset.
- [Residual Network (MNIST)](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_mnist.py). A bottleneck residual network applied to MNIST classification task.
- [Residual Network (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py). A residual network applied to CIFAR-10 classification task.
- [Google Inception (v3)](https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py). Google's Inception v3 network applied to Oxford Flowers 17 classification task.
- [Auto Encoder](https://github.com/tflearn/tflearn/blob/master/examples/images/autoencoder.py). An auto encoder applied to MNIST handwritten digits.

### Natural Language Processing
- [Recurrent Neural Network (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py). Apply an LSTM to IMDB sentiment dataset classification task.
- [Bi-Directional RNN (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/bidirectional_lstm.py). Apply a bi-directional LSTM to IMDB sentiment dataset classification task.
- [Dynamic RNN (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/dynamic_lstm.py). Apply a dynamic LSTM to classify variable length text from IMDB dataset.
- [City Name Generation](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_cityname.py). Generates new US-cities name, using LSTM network.
- [Shakespeare Scripts Generation](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py). Generates new Shakespeare scripts, using LSTM network.
- [Seq2seq](https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py). Pedagogical example of seq2seq reccurent network. See [this repo](https://github.com/ichuang/tflearn_seq2seq) for full instructions.
- [CNN Seq](https://github.com/tflearn/tflearn/blob/master/examples/nlp/cnn_sentence_classification.py). Apply a 1-D convolutional network to classify sequence of words from IMDB sentiment dataset.

### Reinforcement Learning
- [Atari Pacman 1-step Q-Learning](https://github.com/tflearn/tflearn/blob/master/examples/reinforcement_learning/atari_1step_qlearning.py). Teach a machine to play Atari games (Pacman by default) using 1-step Q-learning.

### Others
- [Recommender - Wide & Deep Network](https://github.com/tflearn/tflearn/blob/master/examples/others/recommender_wide_and_deep.py). Pedagogical example of wide & deep networks for recommender systems.

### Notebooks
- [Spiral Classification Problem](https://github.com/tflearn/tflearn/blob/master/examples/notebooks/spiral.ipynb). TFLearn implementation of spiral classification problem from Stanford CS231n.

### Extending TensorFlow
- [Layers](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/layers.py). Use TFLearn layers along with TensorFlow.
- [Trainer](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/trainer.py). Use TFLearn trainer class to train any TensorFlow graph.
- [Built-in Ops](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/builtin_ops.py). Use TFLearn built-in operations along with TensorFlow.
- [Summaries](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/summaries.py). Use TFLearn summarizers along with TensorFlow.
- [Variables](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/variables.py). Use TFLearn variables along with TensorFlow.


## Dependencies
```
tensorflow 1.0alpha
numpy
matplotlib
cuda
tflearn (if using tflearn examples)
```
For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
