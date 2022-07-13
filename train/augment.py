import numpy as np
import tensorflow as tf


def random_crop(tensors, size, seed=None, name=None):
    """Randomly crops multiple tensors (of the same shape) to a given size.

    Each tensor is cropped in the same way."""
    with tf.name_scope(name, "random_crop", [size]) as name:
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        if len(tensors) == 2:
            shape = tf.minimum(tf.shape(tensors[0]), tf.shape(tensors[1]))
        else:
            shape = tf.shape(tensors[0])

        limit = shape - size + 1
        offset = tf.random_uniform(
           tf.shape(shape),
           dtype=size.dtype,
           maxval=size.dtype.max,
           seed=seed) % limit

        results = []
        for tensor in tensors:
            result = tf.slice(tensor, offset, size)
            results.append(result)
        return results
