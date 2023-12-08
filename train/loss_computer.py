from warp import image_warp
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from util import summarized_images
from network import Encoder


def track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))


def normalize(img, mean, stddev):
    return (img - mean) / stddev


def compute_losses(im1, im2, im3, im4, flow, mean, stddev):

    losses = {}

    im4_warped = image_warp(im4, flow)

    losses['gradientSmooth'] = gradientSmooth(flow)

    im1_encoder = normalize(im1, mean, stddev)
    im2_encoder = normalize(im2, mean, stddev)
    im3_encoder = normalize(im3, mean, stddev)
    im4_encoder = normalize(im4, mean, stddev)
    im4_warped_encoder = normalize(im4_warped, mean, stddev)

    feature_f1, im1_recon = Encoder(im1_encoder)
    feature_f2, im2_recon = Encoder(im2_encoder, reuse=True)
    feature_f3, im3_recon = Encoder(im3_encoder, reuse=True)
    feature_m, im4_recon = Encoder(im4_encoder, reuse=True)
    feature_m_warped, im4_warped_recon = Encoder(im4_warped_encoder, reuse=True)

    loss_1 = l2Loss(feature_f1, feature_m_warped)
    loss_2 = l2Loss(feature_f2, feature_m_warped)
    loss_3 = l2Loss(feature_f3, feature_m_warped)

    track_loss(loss_1, 'loss/loss_1')
    track_loss(loss_2, 'loss/loss_2')
    track_loss(loss_3, 'loss/loss_3')

    losses['L2'] = 0.33 * loss_1 + 0.5 * loss_2 + 1.0 * loss_3

    summarized_images(im1, '1st fixed image')
    summarized_images(im2, '2nd fixed image')
    summarized_images(im3, '3rd fixed image')
    summarized_images(im4, 'moving image')
    summarized_images(im4_warped, 'warped moving image')

    summarized_images(im1_recon, '1st fixed image reconstruction')
    summarized_images(im2_recon, '2nd fixed image reconstruction')
    summarized_images(im3_recon, '3rd fixed image reconstruction')
    summarized_images(im4_recon, 'moving image reconstruction')
    summarized_images(im4_warped_recon, 'warped moving image reconstruction')

    summarized_images(feature_f1, '1st fixed image feature')
    summarized_images(feature_f2, '2nd fixed image feature')
    summarized_images(feature_f3, '3rd fixed image feature')
    summarized_images(feature_m, 'moving image feature')
    summarized_images(feature_m_warped, 'warped moving image feature')

    return losses


def gradientSmooth(flow):
    dy = tf.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
    dx = tf.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

    dy = dy * dy
    dx = dx * dx

    d = tf.sqrt(tf.reduce_sum(dx)) + tf.sqrt(tf.reduce_sum(dy))
    return d / 2.0


def l2Loss(feature_1, feature_2):
    # batch, height, width, channels = tf.unstack(tf.shape(feature_1))
    # normalization = tf.cast(batch * height * width * channels, tf.float32)
    loss = tf.sqrt(tf.reduce_sum(tf.square(feature_1 - feature_2)))
    return loss
