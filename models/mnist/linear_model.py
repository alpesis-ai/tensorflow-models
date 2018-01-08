import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    data = input_data.read_data_sets("MNIST_data", one_hot=True)
    print("Size of:")
    print("- training set:\t\t{}".format(len(data.train.labels)))
    print("- test set:\t\t{}".format(len(data.test.labels)))
    print("- validation set:\t{}".format(len(data.validation.labels)))
    return data


def plot_data(data):
    print data.test.labels[0:5, :]
    data.test.cls = np.array([label.argmax() for label in data.test.labels])
    print data.test.cls[0:5]


def plot_images(images, image_shape, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
   
    figure, axes = plt.subplots(3, 3)
    figure.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(image_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_weights(weights, image_shape):
    w_min = np.min(weights)
    w_max = np.max(weights)

    figure, axes = plt.subplots(3, 4)
    figure.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < 10:
            image = weights[:, i].reshape(image_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def train(data, image_size_flatten, num_classes, batch_size, num_iters):
    x = tf.placeholder(tf.float32, [None, image_size_flatten])
    y_true = tf.placeholder(tf.float32, [None, num_classes])
    y_true_cls = tf.placeholder(tf.int64, [None])

    weights = tf.Variable(tf.zeros([image_size_flatten, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # eval
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    # optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for i in range(num_iters):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)


    feed_dict_test = {x: data.test.images,
                      y_true: data.test.labels,
                      y_true_cls: data.test.cls}

    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test set: {0:.1%}".format(acc))  
    
    cls_true = data.test.cls
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # confusion matrix
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plot_confusion_matrix(cm)  
 
    # example errors
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images[0:9], image_shape, cls_true[0:9], cls_pred[0:9])

    # weights
    weights = session.run(weights)
    plot_weights(weights, image_shape) 


if __name__ == '__main__':

    image_height = image_width = 28
    image_size_flatten = image_height * image_width
    image_shape = (image_height, image_width)
    num_classes = 10
    batch_size = 100
    num_iters = 1000

    data = get_data()
    plot_data(data)

    images = data.test.images[0:9]
    cls_true = data.test.cls[0:9]
    plot_images(images, image_shape, cls_true)

    train(data, image_size_flatten, num_classes, batch_size, num_iters)
