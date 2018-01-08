from sklearn.metrics import confusion_matrix
import tensorflow as tf

import settings
from common.data import get_data
from common.eval import eval_confusion_matrix, eval_errors
from common.plot import plot_data, plot_images, plot_weights


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
    eval_confusion_matrix(cls_true, cls_pred, settings.NUM_CLASSES)
 
    # example errors
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    eval_errors(data, image_shape, correct, cls_pred)

    # weights
    weights = session.run(weights)
    plot_weights(weights, image_shape) 


if __name__ == '__main__':

    image_height = settings.IMAGE_HEIGHT
    image_width = settings.IMAGE_WIDTH
    image_size_flatten = settings.IMAGE_SIZE
    image_shape = settings.IMAGE_SHAPE
    num_classes = settings.NUM_CLASSES
    batch_size = settings.BATCH_SIZE
    num_iters = settings.NUM_ITERS

    data = get_data()
    plot_data(data)

    images = data.test.images[0:9]
    cls_true = data.test.cls[0:9]
    plot_images(images, image_shape, cls_true)

    train(data, image_size_flatten, num_classes, batch_size, num_iters)
