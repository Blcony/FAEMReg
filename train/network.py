import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


def leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def regisnet(im1, im2, im3, im4, full_res):
    with tf.variable_scope('RegisNet') as scope:
        inputs = tf.concat([im1, im2, im3, im4, im3, im2, im1], -1)
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        inputs_3D = tf.expand_dims(inputs, -1)

        flows = network(inputs_3D, full_res)

    return flows


def network_upsample(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs, full_res):
    channels = 2

    flow6 = slim.conv3d(conv6_1, channels, 3, stride=[7, 1, 1], scope='flow_6', activation_fn=None)
    flow6_r = flow6[:, 0, :, :, :]
    deconv5 = slim.conv3d_transpose(conv6_1, 256, 3, stride=[1, 2, 2], scope='deconv_5')
    concat5 = tf.concat([conv5_1, deconv5], -1)

    flow5 = slim.conv3d(concat5, channels, 3, stride=[7, 1, 1], scope='flow_5', activation_fn=None)
    flow5_r = flow5[:, 0, :, :, :]
    deconv4 = slim.conv3d_transpose(concat5, 128, 3, stride=[1, 2, 2], scope='deconv_4')
    concat4 = tf.concat([conv4_1, deconv4], -1)

    flow4 = slim.conv3d(concat4, channels, 3, stride=[7, 1, 1], scope='flow_4', activation_fn=None)
    flow4_r = flow4[:, 0, :, :, :]
    deconv3 = slim.conv3d_transpose(concat4, 64, 3, stride=[1, 2, 2], scope='deconv_3')
    concat3 = tf.concat([conv3_1, deconv3], -1)

    flow3 = slim.conv3d(concat3, channels, 3, stride=[7, 1, 1], scope='flow_3', activation_fn=None)
    flow3_r = flow3[:, 0, :, :, :]
    deconv2 = slim.conv3d_transpose(concat3, 32, 3, stride=[1, 2, 2], scope='deconv_2')
    concat2 = tf.concat([conv2, deconv2], -1)

    flow2 = slim.conv3d(concat2, channels, 3, stride=[7, 1, 1], scope='flow_2', activation_fn=None)
    flow2_r = flow2[:, 0, :, :, :]

    flows = [flow2_r, flow3_r, flow4_r, flow5_r, flow6_r]

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv3d_transpose(concat2, 16, 3, stride=[1, 2, 2], scope='deconv_1')
            concat1 = tf.concat([conv1, deconv1], -1)

            flow1 = slim.conv3d(concat1, channels, 3, stride=[7, 1, 1], scope='flow_1', activation_fn=None)
            flow1_r = flow1[:, 0, :, :, :]
            deconv0 = slim.conv3d_transpose(concat1, 8, 3, stride=[1, 2, 2], scope='deconv_0')
            concat0 = tf.concat([inputs, deconv0], -1)

            flow0 = slim.conv3d(concat0, channels, 3, stride=[7, 1, 1], scope='flow_0', activation_fn=None)
            flow0_r = flow0[:, 0, :, :, :]

            flows = [flow0_r, flow1_r] + flows

    return flows


def network(inputs, full_res):

    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        data_format='NDHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=leaky_relu):
        conv1 = slim.conv3d(inputs, 32, [3, 7, 7], stride=[1, 2, 2], scope='conv_1')
        conv2 = slim.conv3d(conv1, 64, [3, 5, 5], stride=[1, 2, 2], scope='conv_2')
        conv3 = slim.conv3d(conv2, 128, [3, 5, 5], stride=[1, 2, 2], scope='conv_3')
        conv3_1 = slim.conv3d(conv3, 128, 3, stride=[1, 1, 1], scope='conv_3_1')
        conv4 = slim.conv3d(conv3_1, 256, 3, stride=[1, 2, 2], scope='conv_4')
        conv4_1 = slim.conv3d(conv4, 256, 3, stride=[1, 1, 1], scope='conv_4_1')
        conv5 = slim.conv3d(conv4_1, 256, 3, stride=[1, 2, 2], scope='conv_5')
        conv5_1 = slim.conv3d(conv5, 256, 3, stride=[1, 1, 1], scope='conv_5_1')
        conv6 = slim.conv3d(conv5_1, 512, 3, stride=[1, 2, 2], scope='conv_6')
        conv6_1 = slim.conv3d(conv6, 512, 3, stride=[1, 1, 1], scope='conv_6_1')

        res = network_upsample(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs, full_res)

    return res


def Encoder(im, reuse=False):
    with tf.variable_scope('Autoencoder'):
        with slim.arg_scope([slim.conv2d],
                            data_format='NHWC',
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            weights_initializer=layers.variance_scaling_initializer()):

            conv1 = slim.conv2d(im, 128,  5, stride=1, scope='encoder/conv1', trainable=False, reuse=reuse)
            conv2 = slim.conv2d(conv1, 64, 3, stride=2, scope='encoder/conv2', trainable=False, reuse=reuse)
            conv3 = slim.conv2d(conv2, 32, 3, stride=2, scope='encoder/conv3', trainable=False, reuse=reuse)
            conv4 = slim.conv2d(conv3, 16, 3, stride=1, scope='encoder/conv4', trainable=False, reuse=reuse)
            conv5 = slim.conv2d(conv4, 8, 3, stride=2, scope='encoder/conv5', trainable=False, reuse=reuse)
            conv6 = slim.conv2d(conv5, 4, 3, stride=1, scope='encoder/conv6', trainable=False, reuse=reuse)

            feature = slim.conv2d(conv6, 1, 3, stride=2, scope='feature', trainable=False, reuse=reuse)

            deconv1 = slim.conv2d_transpose(feature, 4, 3, stride=2, scope='decoder/deconv1', trainable=False, reuse=reuse)
            deconv2 = slim.conv2d_transpose(deconv1, 8, 3, stride=1, scope='decoder/deconv2', trainable=False, reuse=reuse)
            deconv3 = slim.conv2d_transpose(deconv2, 16, 3, stride=2, scope='decoder/deconv3', trainable=False, reuse=reuse)
            deconv4 = slim.conv2d_transpose(deconv3, 32, 3, stride=1, scope='decoder/deconv4', trainable=False, reuse=reuse)
            deconv5 = slim.conv2d_transpose(deconv4, 64, 3, stride=2, scope='decoder/deconv5', trainable=False, reuse=reuse)
            deconv6 = slim.conv2d_transpose(deconv5, 128, 3, stride=2, scope='decoder/deconv6', trainable=False, reuse=reuse)
            im_recon = slim.conv2d_transpose(deconv6, 1, 3, stride=1, scope='im_recon', activation_fn=None, trainable=False, reuse=reuse)

    return feature, im_recon