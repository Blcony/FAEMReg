import os
import sys
import tensorflow as tf
import tifffile as tif
from argparse import ArgumentParser
from data import Data
from util import restore_networks, flow_to_color
from network import regisnet
from image_warp import image_warp
import numpy as np
import time
from functools import cmp_to_key
FLOW_SCALE = 20.0


dims = (1, 640, 640, 1)
mean = 144.603
stddev = 33.378
eval_dir = '/userhome/Data/FIB25/FIB_25_padded_warped'
base_dir = '/userhome/EM_registration/RegisNet/Result_FIB'
ckpt_dir = '/userhome/EM_registration/RegisNet/Model'


def check_path(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


def normalize(img, mean, stddev):
    return (img - mean) / stddev


def write_tif(z, path):
    z = z[0, :, :, 0]
    out = z.astype(np.uint8)
    tif.imsave(path, out)


def write_flow(z, path):
    tif.imsave(path, z)


def write_rgb_png(z, path):
    z = z[0, :, :, :]
    out = z.astype(np.uint8)
    tif.imsave(path, out)


def evaluate(name, data, ckpt_iter, full_res=True):
    output_dir = check_path(os.path.join(base_dir, name, 'output_' + ckpt_iter))

    ckpt_path = os.path.join(ckpt_dir, name, 'ckpt')
    print("ckpt_path: ", ckpt_path)
    ckpt = ckpt_path + '/model.ckpt-' + ckpt_iter

    image_fixed_1 = tf.placeholder(tf.float32, shape=dims)
    image_fixed_2 = tf.placeholder(tf.float32, shape=dims)
    image_fixed_3 = tf.placeholder(tf.float32, shape=dims)
    image_moving = tf.placeholder(tf.float32, shape=dims)

    image_fixed1_nom = normalize(image_fixed_1, mean, stddev)
    image_fixed2_nom = normalize(image_fixed_2, mean, stddev)
    image_fixed3_nom = normalize(image_fixed_3, mean, stddev)
    image_moving_nom = normalize(image_moving, mean, stddev)

    flows = regisnet(image_fixed1_nom, image_fixed2_nom, image_fixed3_nom, image_moving_nom, full_res=full_res)
    flow = flows[0] * FLOW_SCALE

    image_warped = image_warp(image_moving, flow)

    image_slots = [(image_fixed_3, 'image_fixed_pad'),
                   (image_moving, 'image_moving_pad'),
                   (image_warped, 'image_warped'),
                   (flow[0, :, :, 0], 'flow prediction 1'),
                   (flow[0, :, :, 1], 'flow prediction 1'),
                   (flow_to_color(flow), 'flow prediction')]

    num_ims = len(image_slots)
    image_ops = [t[0] for t in image_slots]

    sess_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        restore_networks(sess, ckpt)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        max_iter = args.num

        filenames = data.get_warped_filenames()
        image_1 = tif.imread(filenames[0])
        image_1 = np.reshape(image_1, dims)
        image_2 = tif.imread(filenames[0])
        image_2 = np.reshape(image_2, dims)
        image_3 = tif.imread(filenames[0])
        image_3 = np.reshape(image_3, dims)

        while not coord.should_stop():
            Duration = []
            for i in range(max_iter):
                image_4 = tif.imread(filenames[i + 1])
                image_4 = np.reshape(image_4, dims)

                start = time.time()
                [result] = sess.run([image_ops], feed_dict={image_fixed_1: image_1, image_fixed_2: image_2, image_fixed_3: image_3, image_moving: image_4})
                Duration.append(time.time() - start)
                image_results = result[:num_ims]
                iterstr = str(i).zfill(6)

                path_imgFixed = os.path.join(output_dir, iterstr + '_imgFixed.tif')
                path_imgMoving = os.path.join(output_dir, iterstr + '_imgMoving.tif')
                path_imgWarped = os.path.join(output_dir, iterstr + '_imgWarped.tif')
                path_flow1 = os.path.join(output_dir, iterstr + '_flow1.tif')
                path_flow2 = os.path.join(output_dir, iterstr + '_flow2.tif')
                path_flow = os.path.join(output_dir, iterstr + '_flow.tif')

                write_tif(image_results[0], path_imgFixed)
                write_tif(image_results[1], path_imgMoving)
                write_tif(image_results[2], path_imgWarped)
                write_flow(image_results[3], path_flow1)
                write_flow(image_results[4], path_flow2)
                write_rgb_png(image_results[5] * 255, path_flow)

                image_1 = image_2
                image_2 = image_3
                image_3 = image_results[2]

                sys.stdout.write("-- evaluate '{}': {}/{} image pairs".format(name, i, max_iter - 1))
                sys.stdout.flush()
                print()

            np.savetxt(os.path.join(output_dir, 'average_time.txt'), np.array(Duration))
            break

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, dest="model", help="Name of model to eval")
    parser.add_argument("--ckpt", type=str, required=True, dest="ckpt_iter", help="Number of iterations")
    parser.add_argument("--num", type=int, dest="num", default=519, help="Number of examples to evaluate")
    parser.add_argument("--gpu", type=int, dest="gpu", default=0, help="GPU device to evaluate on")
    args = parser.parse_args()

    print("--evaluating on {} pairs from {}".format(args.num, args.model))

    for name in args.model.split(','):
        save_dir = check_path(os.path.join(base_dir, name))
        data = Data(eval_dir=eval_dir)
        evaluate(name, data, args.ckpt_iter)
