import tensorflow as tf

@tf.function
def add(p, q):
    return p + q


print(add(tf.constant([1, 2, 3, 4]), tf.constant([5, 6, 7, 8])))