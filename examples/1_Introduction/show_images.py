from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./data/", one_hot=True)

sn = 1000
amount = 20
lines = 4
columns = 5
image = np.zeros((amount, 28, 28))
number = np.zeros(amount)

for i in range(amount):
    image[i] = mnist.train.images[sn + i].reshape(28, 28)
    label = mnist.train.labels[sn + i]
    number[i] = int(np.where(label == 1)[0])
    # print(number[0])

fig = plt.figure()

for i in range(amount):
    ax = fig.add_subplot(lines, columns, 1 + i)
    plt.imshow(image[i], cmap='binary')
    plt.sca(ax)

plt.xlabel(number)
plt.show()
