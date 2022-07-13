from util import summarized_placeholder
from loss import Loss
import tensorflow.contrib.slim as slim
import tensorflow as tf
import os
import re
epoch = 197
autoencoder_ckpt_dir = '/userhome/EM_registration/Autoencoder/Model/modelRaw7-15_1/ckpt'


class Trainer():
    def __init__(self, train_batch_fn, params, normalization,
                 train_summaries_dir, ckpt_dir, loss_dir, devices=None):
        self.train_batch_fn = train_batch_fn
        self.params = params
        self.normalization = normalization
        self.train_summaries_dir = train_summaries_dir
        self.ckpt_dir = ckpt_dir
        self.loss_dir = loss_dir
        self.devices = devices[0]
        self.loss_fn = Loss

    def restore_networks(self, sess, ckpt):
        net_names = ['RegisNet']
        variables_to_save = slim.get_variables_to_restore(include=net_names)
        saver = tf.train.Saver(variables_to_save, max_to_keep=50)
        sess.run(tf.global_variables_initializer())

        if ckpt is not None:
            print('Continue training:', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

        ckpt_autoencoder = tf.train.get_checkpoint_state(autoencoder_ckpt_dir)
        print('Autoencoder ckpt path: ', ckpt_autoencoder.model_checkpoint_path)
        nets_to_restore = ['Autoencoder']
        variables_to_restore = slim.get_variables_to_restore(include=nets_to_restore)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, ckpt_autoencoder.model_checkpoint_path)

        return saver

    def add_summaries(self):
        self.add_loss_summaries()
        self.add_param_summaries()

    def add_loss_summaries(self):
        losses = tf.get_collection('losses')
        for l in losses:
            tensor_name = re.sub('tower_[0-9]*/', '', l.op.name)
            tf.summary.scalar(tensor_name, l)

    def add_param_summaries(self):
        params = tf.get_collection('params')
        for p in params:
            tensor_name = re.sub('tower_[0-9]*/', '', p.op.name)
            tf.summary.scalar(tensor_name, p)

    def get_train_and_loss_ops(self, batch, learning_rate, global_step):
        opt = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999, learning_rate=learning_rate)
        loss_ = self.loss_fn(batch, self.normalization)
        train_op = opt.minimize(loss_)
        self.add_summaries()
        return train_op, loss_

    def run(self, min_iter, max_iter):
        save_interval = self.params['save_iter']
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt is not None:
            ckpt_path = ckpt.model_checkpoint_path
            global_step = int(ckpt_path.split('/')[-1].split('-')[-1])
            assert global_step >= min_iter, 'training stage not reached'

            start_iter = global_step + 1
            if start_iter >= max_iter:
                print('-- train: max_iter reached')
                return
        else:
            start_iter = min_iter + 1

        print('-- training from i = {} to {}'.format(start_iter, max_iter))

        assert (max_iter - start_iter + 1) % save_interval == 0

        for i in range(start_iter, max_iter + 1, save_interval):
            self.train(i, i + save_interval - 1, i - (min_iter + 1))

    def train(self, start_iter, max_iter, iter_offset):

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)

        with tf.Graph().as_default():
            batch = self.train_batch_fn(iter_offset)

            with tf.name_scope('param'):
                learning_rate_ = summarized_placeholder('learning', 'train')

            global_step_ = tf.placeholder(tf.int32, name='global_step')

            train_op, loss_ = self.get_train_and_loss_ops(batch, learning_rate_, global_step_)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summary_ = tf.summary.merge(summaries)

            sess_config = tf.ConfigProto(allow_soft_placement=True)

            with tf.Session(config=sess_config) as sess:
                summary_writer = tf.summary.FileWriter(self.train_summaries_dir)

                saver = self.restore_networks(sess, ckpt)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                for local_i, i in enumerate(range(start_iter, max_iter + 1)):
                    decay_iters = local_i + iter_offset
                    decay_interval = self.params['decay_interval']
                    decay_after = self.params['decay_after']

                    if decay_iters >= decay_after:
                        decay_mini = decay_after / decay_interval
                        decay = (decay_iters // decay_interval) - decay_mini
                        learning_rate = self.params['lr'] / (2 ** decay)
                    else:
                        learning_rate = self.params['lr']

                    feed_dict = {learning_rate_: learning_rate, global_step_: i}
                    _, loss = sess.run([train_op, loss_], feed_dict=feed_dict)

                    if i == 1 or i % self.params['display_iter'] == 0:
                        summary = sess.run(summary_, feed_dict=feed_dict)
                        summary_writer.add_summary(summary, i)
                        print("-- train: i= {}, loss={}".format(i, loss))

                save_path = os.path.join(self.ckpt_dir, 'model.ckpt')
                saver.save(sess, save_path, global_step=max_iter)

                summary_writer.close()
                coord.request_stop()
                coord.join(threads)
