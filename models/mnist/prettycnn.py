import tensorflow as tf
import prettytensor as pt

import settings
from common.data import get_data


def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


def model():
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty \
                       .conv2d(kernel=5, depth=16, name="layer_conv1") \
                       .max_pool(kernel=2, stride=2) \
                       .conv2d(kernel=5, depth=36, name="layer_conv2") \
                       .max_pool(kernel=2, stride=2) \
                       .flatten() \
                       .fully_connected(size=128, name="layer_fc1") \
                       .softmax_classifier(num_classes=settings.NUM_CLASSES, labels=y_true)
    return y_pred, loss



def train():
    y_pred, loss = model()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # train
    iterx = 0;
    for i in range(iterx, settings.NUM_ITERS):
        x_batch, y_true_batch = data.train.next_batch(settings.BATCH_SIZE)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i+1, acc))
    iterx += settings.NUM_ITERS


if __name__ == '__main__':

    data = get_data()

    x = tf.placeholder(tf.float32, shape=[None, settings.IMAGE_SIZE], name='x')
    x_image = tf.reshape(x, [-1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANNELS]) 
    x_pretty = pt.wrap(x_image)
    y_true = tf.placeholder(tf.float32, shape=[None, settings.NUM_CLASSES], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)


    train()


    weights_conv1 = get_weights_variable(layer_name='layer_conv1')
    weights_conv2 = get_weights_variable(layer_name='layer_conv2')
    weights_fc1 = get_weights_variable(layer_name='layer_fc1')
