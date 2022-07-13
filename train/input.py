import os
import random
import numpy as np
import tifffile as tif
from functools import cmp_to_key
import tensorflow as tf
from augment import random_crop
raw_shape = [3072, 3072, 1]


class Input():
    def __init__(self, data, batch_size, dims, num_thread=1, skipped_frames=False):
        assert len(dims) == 2
        self.data = data
        self.dims = dims
        self.batch_size = batch_size
        self.num_threads = num_thread
        self.skipped_frames = skipped_frames
        self.mean = 144.603
        self.stddev = 33.378

    def get_normalization(self):
        return self.mean, self.stddev

    def imread(self, tensor):
        path = tensor
        final_path = bytes.decode(path)
        image = tif.imread(final_path)
        image = image.astype(np.float32)
        return image

    def read_tif_image(self, file):
        [file_queue] = tf.train.slice_input_producer([file], shuffle=False)
        image = tf.py_func(lambda x: self.imread(x), [file_queue], tf.float32)
        image = tf.expand_dims(image, -1)
        image = tf.reshape(image, raw_shape)
        return image

    def input(self, needs_crop=True, shift=0, seed=0, center_crop=False, swap_images=False):

        data_dirs = self.data.get_raw_dirs()
        height, width = self.dims

        filenames = []
        for dir_path in data_dirs:
            files = os.listdir(dir_path)
            files.sort(key=cmp_to_key(lambda x, y: (int(x.split('.')[0]) - int(y.split('.')[0]))))
            length = len(files)
            for i in range(length - 3):
                fn1 = os.path.join(dir_path, files[i])
                fn2 = os.path.join(dir_path, files[i + 1])
                fn3 = os.path.join(dir_path, files[i + 2])
                fn4 = os.path.join(dir_path, files[i + 3])
                filenames.append((fn1, fn2, fn3, fn4))

        random.seed(seed)
        random.shuffle(filenames)
        print("Training on {} images".format(len(filenames)))

        filenames_extended = []
        for fn1, fn2, fn3, fn4 in filenames:
            filenames_extended.append((fn1, fn2, fn3, fn4))
            if swap_images:
                filenames_extended.append((fn4, fn3, fn2, fn1))

        shift = shift % len(filenames)
        filenames_extended = list(np.roll(filenames_extended, shift))

        filenames_1, filenames_2, filenames_3, filenames_4 = zip(*filenames_extended)
        filenames_1 = list(filenames_1)
        filenames_2 = list(filenames_2)
        filenames_3 = list(filenames_3)
        filenames_4 = list(filenames_4)

        with tf.variable_scope('train_inputs'):
            image_1 = self.read_tif_image(filenames_1)
            image_2 = self.read_tif_image(filenames_2)
            image_3 = self.read_tif_image(filenames_3)
            image_4 = self.read_tif_image(filenames_4)

            if needs_crop:
                if center_crop:
                    image_1 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                    image_2 = tf.image.resize_image_with_crop_or_pad(image_2, height, width)
                    image_3 = tf.image.resize_image_with_crop_or_pad(image_3, height, width)
                    image_4 = tf.image.resize_image_with_crop_or_pad(image_4, height, width)
                else:
                    image_1, image_2, image_3, image_4 = random_crop([image_1, image_2, image_3, image_4], [height, width, 1])
            else:
                image_1 = tf.reshape(image_1, [height, width, 1])
                image_2 = tf.reshape(image_2, [height, width, 1])
                image_3 = tf.reshape(image_3, [height, width, 1])
                image_4 = tf.reshape(image_4, [height, width, 1])

            return tf.train.batch([image_1, image_2, image_3, image_4],
                                  batch_size=self.batch_size,
                                  num_threads=self.num_threads)
