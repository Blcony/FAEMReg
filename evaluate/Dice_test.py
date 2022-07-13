import os
from os.path import join
import tifffile as tif
import numpy as np
from collections import Counter
from argparse import ArgumentParser


def check_path(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


def Dice_cal(fixed, moving):
    top = 2 * np.sum(fixed * moving)
    bottom = np.maximum(np.sum(fixed + moving), 1e-5)
    Dice = top / bottom
    return Dice


def Dice_greedy(srcpath_fixed, srcpath_moving, weighted=False, neuron_number=50):
    slice_number = 32

    label_fixed = tif.imread(srcpath_fixed)[1:slice_number]

    shape1 = neuron_number
    image_number = label_fixed.shape[0]

    Dice = np.zeros(shape=[shape1, image_number])
    Weight = np.zeros(shape=[shape1, image_number])
    neuron_index = []

    fixed_tile = np.reshape(label_fixed, [-1])
    fixed_l = fixed_tile.tolist()
    count = Counter(fixed_l).most_common(neuron_number + 1)
    print(count[0][0])
    count.pop(0)

    for j in range(neuron_number):
        n = count[j][0]
        neuron_index.append(n)

        fixed = tif.imread(srcpath_fixed)[1: slice_number]
        moving = tif.imread(srcpath_moving)[1: slice_number]

        neurons_num = count[j][1]

        for i in range(image_number):
            Slice_f = fixed[i]
            Slice_f[Slice_f != n] = 0
            neuron_f = Slice_f / n
            neuron_num = np.sum(neuron_f)
            weight = neuron_num / neurons_num
            if weighted:
                Weight[j][i] = weight
            else:
                Weight[j][i] = np.sign(weight)
            Slice_m = moving[i]
            Slice_m[Slice_m != n] = 0
            neuron_m = Slice_m / n

            if neuron_num == 0:
                Dice[j][i] = 0.0
                continue
            dice = Dice_cal(neuron_f, neuron_m)
            Dice[j][i] = dice

    neuron_dice = np.sum(Dice * Weight, 1)
    if weighted:
        dice = neuron_dice
        dice_final = np.mean(dice)
        for i in range(dice.shape[0]):
            print('Neuron id:', neuron_index[i], 'dice: ', dice[i])
        print('Final dice: ', dice_final)
    else:
        dice_n = 1 / np.sum(Weight, 1) * neuron_dice
        dice_n_final = np.mean(dice_n)
        for i in range(dice_n.shape[0]):
            print('Neuron id:', neuron_index[i], 'dice: ', dice_n[i])
        print('Final dice: ', dice_n_final)

        image_dice = np.sum(Dice * Weight, 0)
        dice_i = 1 / np.sum(Weight, 0) * image_dice
        for i in range(dice_i.shape[0]):
            print(i, 'th image dice: ', dice_i[i])
        dice_i_final = np.mean(dice_i)
        dice_i_final_std = np.std(dice_i)
        print('Final dice: ', dice_i_final, dice_i_final_std)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, dest="model", help="Name of model to eval")
    parser.add_argument("--ckpt", type=str, required=True, dest="ckpt_iter", help="Number of iterations")
    args = parser.parse_args()

    Warped_label_path = '/userhome/EM_registration/RegisNet/Result_test/' + args.model + '/label_warped_' + args.ckpt_iter + '.tif'

    Dice_greedy('/userhome/Data/CremiA_1180/expand/raw/test/test_label_expand_padded.tif',
                Warped_label_path,
                weighted=False)
