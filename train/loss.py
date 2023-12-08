import tensorflow as tf
from network import regisnet
from loss_computer import compute_losses
from ops import downsample
FLOW_SCALE = 20.0
LOSSES = ['L2', 'gradientSmooth']
LOSSES_WEIGHT = {'L2_weight': 1.0, 'gradientSmooth_weight': 3.0}


def normalize(img, mean, stddev):
    return (img - mean) / stddev


def track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))


def Loss(batch, normalization=None, augment=False, full_res=True, pyramid_loss=False, return_flow=False):
    mean = tf.constant(normalization[0])
    stddev = tf.constant(normalization[1])

    im1, im2, im3, im4 = batch
    im_shape = tf.shape(im1)[1:3]

    if augment:
##############################################  something should be determined  #######################################
        im1_photo = normalize(im1, mean, stddev)
        im2_photo = normalize(im2, mean, stddev)
        im3_photo = normalize(im3, mean, stddev)
        im4_photo = normalize(im4, mean, stddev)
        print('Empty')

    else:
        im1_photo = normalize(im1, mean, stddev)
        im2_photo = normalize(im2, mean, stddev)
        im3_photo = normalize(im3, mean, stddev)
        im4_photo = normalize(im4, mean, stddev)

    im1_input = im1_photo
    im2_input = im2_photo
    im3_input = im3_photo
    im4_input = im4_photo

    flows = regisnet(im1_input, im2_input, im3_input, im4_input, full_res=full_res)

    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]

    if full_res:
        layer_weights = [12.7, 5.5, 5.0, 4.35, 3.9, 3.4, 1.1]
        im1_s = im1
        im2_s = im2
        im3_s = im3
        im4_s = im4
        final_flow_scale = FLOW_SCALE
        final_flow = flows[0] * FLOW_SCALE
    else:
        im1_s = downsample(im1, 4)
        im2_s = downsample(im2, 4)
        im3_s = downsample(im3, 4)
        im4_s = downsample(im4, 4)
        final_flow_scale = FLOW_SCALE / 4.0
        final_flow = tf.image.resize_bilinear(flows[0], im_shape) * FLOW_SCALE

    if pyramid_loss:
        flows_enum = enumerate(flows)
    else:
        flows_enum = [(0, flows[0])]

    losses_combined = 0.0
    losses_diff = dict()

    for loss in LOSSES:
        losses_diff[loss] = 0.0

    for i, flow in flows_enum:

        layer_name = "loss" + str(i + 2)

        flow_scale = final_flow_scale / (2**i)

        losses = compute_losses(im1_s, im2_s, im3_s, im4_s, flow * flow_scale, mean, stddev)

        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]

            layer_loss = 0.0

            for loss in LOSSES:
                weight_name = loss + '_weight'
                track_loss(losses[loss], loss)
                layer_loss += LOSSES_WEIGHT[weight_name] * losses[loss]
                losses_diff[loss] += layer_weight * losses[loss]

            losses_combined += layer_weight * layer_loss

            im1_s = downsample(im1_s, 2)
            im2_s = downsample(im2_s, 2)
            im3_s = downsample(im3_s, 2)
            im4_s = downsample(im4_s, 2)
    track_loss(losses_combined, 'loss/losses_combined')

    regularization_loss = tf.losses.get_regularization_loss()
    track_loss(regularization_loss, 'loss/reg_loss')

    final_loss = losses_combined + regularization_loss
    track_loss(final_loss, 'loss/final_loss')

    for loss in LOSSES:
        track_loss(losses_diff[loss], 'loss/' + loss)
        weight_name = loss + '_weight'
        weight = tf.identity(LOSSES_WEIGHT[weight_name], name='weight/' + loss)
        tf.add_to_collection('param', weight)

    if not return_flow:
        return final_loss

    return final_loss, final_flow

















