import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def summarized_weights(weight, name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    tf.summary.histogram(prefix + name, weight, collections=[key])


def summarized_images(image, name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    tf.summary.image(prefix + name, image, collections=[key])


def atan2(y, x):
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.greater_equal(y,0.0)),
                      tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.less(y,0.0)),
                      tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)),
                      np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)),
                      -np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0),tf.equal(y,0.0)),
                      np.nan * tf.zeros_like(x), angle)
    return angle


def flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.

    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    """
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(max_flow, 1)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask


def restore_networks(sess, ckpt):
    net_names = ['RegisNet']
    variables_to_save = slim.get_variables_to_restore(include=net_names)
    saver = tf.train.Saver(variables_to_save, max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    if ckpt is not None:
        print('Ckpt path exits:', ckpt)
        saver.restore(sess, ckpt)

    return saver
