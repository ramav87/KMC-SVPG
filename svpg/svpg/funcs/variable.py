import tensorflow as tf
# tf.set_random_seed(1)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.2, shape=shape)
    return tf.Variable(initial)