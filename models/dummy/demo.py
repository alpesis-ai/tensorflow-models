import tensorflow as tf

def data():
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    print("node1, node2: ", node1, node2)
    return node1, node2


def ops(node1, node2):
    node3 = tf.add(node1, node2)
    print("node3: ", node3)
    return node3


def add_ops():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    return adder_node, a, b


if __name__ == '__main__':

    session = tf.Session()

    # constant nodes
    node1, node2 = data()
    print("node1, node2: ", session.run([node1, node2]))
    node3 = ops(node1, node2)
    print("node3: ", session.run(node3))
  
    # custom nodes
    add_ops, a, b = add_ops() 
    print("a+b: ", session.run(add_ops, {a: 3, b: 4.5}))
    print("a+b: ", session.run(add_ops, {a: [1, 3], b:[2, 4]}))
    add_and_triple = add_ops * 3.
    print("(a+b)*3:  ", session.run(add_and_triple, {a: 3, b: 4}))
    print("(a+b)*3: ", session.run(add_and_triple, {a: [1, 3], b: [2, 4]}))

    # trainable params with initial value
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)

    # model
    linear_model = W * x + b
    init = tf.global_variables_initializer()
    session.run(init)
    print("linear model: ", session.run(linear_model, {x: [1, 2, 3, 4]}))

    # eval
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print("loss: ", session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # update trainable params
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    session.run([fixW, fixb])
    print("fixed loss: ", session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
