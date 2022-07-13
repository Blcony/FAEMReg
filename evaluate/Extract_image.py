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

    Flow_path = '/userhome/EM_registration/RegisNet/Result/' + args.model + '/output_' + args.ckpt_iter

    Warped_image_path = check_path(
        '/userhome/EM_registration/RegisNet/Result/' + args.model + '/image_warped_' + args.ckpt_iter)

    Extract_warpedimage(Flow_path,
                        Warped_image_path)


