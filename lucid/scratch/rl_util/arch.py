import tensorflow as tf


def clear_cnn(unscaled_images, batch_norm=False):
    """A simple convolutional architecture designed with interpretability in
    mind:

    - Later convolutional layers have been replaced with dense layers, to allow
        for non-visual processing
    - There are no residual connections, so that the flow of information passes
        through every layer
    - A pool size equal to the stride has been used, to avoid gradient gridding
    - L2 pooling has been used instead of max pooling, for more continuous
        gradients

    Batch norm has been optionally included to help with optimization.
    """

    def conv_layer(out, filters, kernel_size):
        out = tf.compat.v1.layers.conv2d(
            out, filters, kernel_size, padding="same", activation=None
        )
        if batch_norm:
            out = tf.compat.v1.layers.batch_normalization(out)
        out = tf.nn.relu(out)
        return out

    def pool_l2(out, pool_size):
        return tf.sqrt(
            tf.compat.v1.layers.average_pooling2d(
                out ** 2, pool_size=pool_size, strides=pool_size, padding="same"
            )
            + 1e-8
        )

    out = tf.cast(unscaled_images, tf.float32) / 255.0
    with tf.compat.v1.variable_scope("1a"):
        out = conv_layer(out, 16, 7)
        out = pool_l2(out, 2)
    with tf.compat.v1.variable_scope("2a"):
        out = conv_layer(out, 32, 5)
    with tf.compat.v1.variable_scope("2b"):
        out = conv_layer(out, 32, 5)
        out = pool_l2(out, 2)
    with tf.compat.v1.variable_scope("3a"):
        out = conv_layer(out, 32, 5)
        out = pool_l2(out, 2)
    with tf.compat.v1.variable_scope("4a"):
        out = conv_layer(out, 32, 5)
        out = pool_l2(out, 2)
    out = tf.compat.v1.layers.flatten(out)
    out = tf.compat.v1.layers.dense(out, 256, activation=tf.nn.relu)
    out = tf.compat.v1.layers.dense(out, 512, activation=tf.nn.relu)
    return out
