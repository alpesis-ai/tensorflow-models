import tensorflow as tf

import settings
from common.data import get_data
from common.eval import eval_confusion_matrix, eval_errors


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
   
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights
    

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def model(x_image, num_classes):
    layer1_channels = 1
    layer1_filter_size = 3
    layer1_num_filters = 16
    layer2_channels = layer1_num_filters
    layer2_filter_size = 3
    layer2_num_filters = 32 
    fc_size = 128

    layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                                num_input_channels=layer1_channels,
                                                filter_size=layer1_filter_size,
                                                num_filters=layer1_num_filters,
                                                use_pooling=True)
    
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                                num_input_channels=layer2_channels,
                                                filter_size=layer2_filter_size,
                                                num_filters=layer2_num_filters,
                                                use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)

    return layer_fc2


def train(data, image_shape, x_image, y_true):
    result = model(x_image, settings.NUM_CLASSES)
    y_pred = tf.nn.softmax(result)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
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

    # test
    feed_dict_test = {x: data.test.images,
                      y_true: data.test.labels,
                      y_true_cls: data.test.cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test set: {0:.1%}".format(acc))

    cls_true = data.test.cls
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # confusion matrix
    eval_confusion_matrix(cls_true, cls_pred, settings.NUM_CLASSES)

    # plot errors
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    eval_errors(data, image_shape, correct, cls_pred) 

    session.close()


if __name__ == '__main__':

    data = get_data()
 
    x = tf.placeholder(tf.float32, shape=[None, settings.IMAGE_SIZE], name='x')
    x_image = tf.reshape(x, [-1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANNELS])
    y_true = tf.placeholder(tf.float32, shape=[None, settings.NUM_CLASSES], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    train(data, settings.IMAGE_SHAPE, x_image, y_true)
