import tifffile as tif
import numpy as np
import os
from os.path import join
import tensorflow as tf
from label_warp import label_warp
from functools import cmp_to_key
from argparse import ArgumentParser


def check_path(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


def Slice_to_Volume(srcpath=None, dstpath=None):
    f = [l
         for l in os.listdir(srcpath)
         if l.split('.')[-1] == 'tif']
    f.sort(key=cmp_to_key(lambda x, y: int(x.split('.')[0]) - int(y.split('.')[0])))
    Volume = []

    for i in f:
        Slice = tif.imread(join(srcpath, i))
        Slice = Slice.reshape([1344, 1344])
        Volume.append(Slice)

    tif.imsave(dstpath, np.array(Volume))


def Warp_greedy(Filelist, Flow1, Flow2, Warped_label_path):

    image_reference = tif.imread(join(Label_path, Filelist[0]))
    tif.imsave(Warped_label_path + '/' + str(0) + '.tif', image_reference)

    for i in range(0, 61):
        Moving_image = tif.imread(join(Label_path, Filelist[i + 1]))
        Moving_image = tf.cast(tf.expand_dims(tf.expand_dims(Moving_image, -1), 0), tf.float32)
        flow1 = tif.imread(join(Flow_path, Flow1[i]))
        flow1 = tf.cast(tf.expand_dims(tf.expand_dims(flow1, -1), 0), tf.float32)
        flow2 = tif.imread(join(Flow_path, Flow2[i]))
        flow2 = tf.cast(tf.expand_dims(tf.expand_dims(flow2, -1), 0), tf.float32)

        flow = tf.concat([flow1, flow2], -1)

        Warped_image = label_warp(Moving_image, flow)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            [image_warped] = sess.run([Warped_image])
            image_warped = np.reshape(image_warped.astype(np.int32), (1344, 1344))
            tif.imsave(Warped_label_path + '/' + str(i + 1) + '.tif', image_warped)


def Extract_warpedimage(srcpath, dstpath):
    ls = [f
          for f in os.listdir(srcpath)
          if f.split('.')[0].split('_')[1] == 'imgWarped']
    image_reference = tif.imread(join(srcpath, '000000_imgFixed.tif'))
    tif.imsave(join(dstpath, '000000_imgFixed.tif'), image_reference)
    for i in ls:
        image = tif.imread(join(srcpath, i))
        tif.imsave(join(dstpath, i), image)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, dest="model", help="Name of model to eval")
    parser.add_argument("--ckpt", type=str, required=True, dest="ckpt_iter", help="Number of iterations")
    args = parser.parse_args()

    Flow_path = '/userhome/EM_registration/RegisNet/Result_test_62/' + args.model + '/output_' + args.ckpt_iter

    Label_path = '/userhome/Data/CremiA_1180/expand/warp/test_1_label'

    Warped_label_path = check_path(
        '/userhome/EM_registration/RegisNet/Result_test_62/' + args.model + '/label_warped_' + args.ckpt_iter)

    Warped_image_path = check_path(
        '/userhome/EM_registration/RegisNet/Result_test_62/' + args.model + '/image_warped_' + args.ckpt_iter)

    Filelist = [f
                for f in os.listdir(Label_path)]
    Flow1 = [f
             for f in os.listdir(Flow_path)
             if f.split('_')[1][0:5] == 'flow1']
    Flow2 = [f
             for f in os.listdir(Flow_path)
             if f.split('_')[1][0:5] == 'flow2']

    Filelist.sort(key=cmp_to_key(lambda x, y: int(x.split('.')[0]) - int(y.split('.')[0])))
    Flow1.sort(key=cmp_to_key(lambda x, y: int(x.split('_')[0]) - int(y.split('_')[0])))
    Flow2.sort(key=cmp_to_key(lambda x, y: int(x.split('_')[0]) - int(y.split('_')[0])))

    Warp_greedy(Filelist, Flow1, Flow2, Warped_label_path)

    Extract_warpedimage(Flow_path,
                        Warped_image_path)

    Slice_to_Volume(Warped_label_path,
                    Warped_label_path + '.tif')

