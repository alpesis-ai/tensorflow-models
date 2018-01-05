import tensorflow as tf


def save():
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    saver = tf.train.Saver([w1, w2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "./data/model1/model", global_step=1000)


def restore():
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph('./data/model1/model-1000.meta')
      saver.restore(sess, tf.train.latest_checkpoint('./data/model1'))
      print(sess.run('w1:0'))


def save_model():
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1 = tf.Variable(2.0, name="bias")
    feed_dict = {w1: 4, w2: 8}

    w3 = tf.add(w1, w2)
    w4 = tf.multiply(w3, b1, name="op_to_restore")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print(sess.run(w4, feed_dict))
    saver.save(sess, './data/model2/test_model', global_step=1000)


def restore_model():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./data/model2/test_model-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./data/model2'))

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name('w1:0')
    w2 = graph.get_tensor_by_name('w2:0')
    feed_dict = {w1: 13.0, w2: 17.0}

    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    print(sess.run(op_to_restore, feed_dict))

    add_on_op = tf.multiply(op_to_restore, 2)
    print(sess.run(add_on_op, feed_dict))


if __name__ == '__main__':
    # save()
    # restore()
    save_model()
    restore_model()
