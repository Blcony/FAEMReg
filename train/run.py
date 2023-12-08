from argparse import ArgumentParser
from train import Trainer
from input import Input
from data import Data
import os
from os.path import join

size = (1024, 1024)

data_dir = '/userhome/Data/Fafb/Warped'
base_dir = '/userhome/EM_registration/RegisNet/Model'


def check_path(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, dest="model", help='Name of model when saving')
    parser.add_argument("--gpu", type=int, default=0, dest="gpu_id", help="Gpu id number")
    parser.add_argument("--lr", type=float, dest="lr", default=1.0e-4, help="learning rate")
    parser.add_argument("--iters", type=int, dest="n_iterations", default=197000, help="number of iterations")
    parser.add_argument("--lambda", type=float, dest="reg_param", default=1.0, help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int, dest="model_save_iter", default=3940, help="frequency of model saves")
    parser.add_argument("--display_iter", type=int, dest="display_iter", default=197, help="frequency of loss displays")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="batch_size for training")
    parser.add_argument("--decay_after", type=int, dest="decay_after", default=49250, help="iteration threshold for decreasing learning rate")
    parser.add_argument("--decay_interval", type=int, dest="decay_interval", default=49250, help="frequency of learning rate decreases")
    args = parser.parse_args()

    model_dir = check_path(join(base_dir, args.model))
    train_dir = check_path(join(model_dir, 'train'))
    ckpt_dir = check_path(join(model_dir, 'ckpt'))
    loss_dir = check_path(join(model_dir, 'loss'))

    devices = ['/gpu:' + str(args.gpu_id)]
    params = {"lr": args.lr, "reg_param": args.reg_param,
              "save_iter": args.model_save_iter,
              "decay_after": args.decay_after,
              "decay_interval": args.decay_interval,
              "display_iter": args.display_iter}

    data = Data(data_dir=data_dir)
    einput = Input(data=data,
                   batch_size=args.batch_size,
                   dims=size)
    iters = args.n_iterations
    tr = Trainer(
            lambda shift: einput.input(shift=shift * args.batch_size),
            params=params,
            normalization=einput.get_normalization(),
            train_summaries_dir=train_dir,
            ckpt_dir=ckpt_dir,
            loss_dir=loss_dir,
            devices=devices)
    tr.run(0, iters)
