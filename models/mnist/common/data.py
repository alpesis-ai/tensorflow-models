import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    data = input_data.read_data_sets("MNIST_data", one_hot=True)
    print("Size of:")
    print("- training set:\t\t{}".format(len(data.train.labels)))
    print("- test set:\t\t{}".format(len(data.test.labels)))
    print("- validation set:\t{}".format(len(data.validation.labels)))
    data.test.cls = np.argmax(data.test.labels, axis=1)
    return data
