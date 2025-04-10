import os
import json
import argparse
from datetime import datetime
import torch


def print_arguments(args):
    arg_list = sorted(vars(args).items())
    for key, value in arg_list:
        print('{}: {}'.format(key, value))


def write_arguments_file(filename, args, sort=True):
    if not isinstance(args, dict):
        args = vars(args)
    with open(filename, 'w') as f:
        json.dump(args, f, indent=4, sort_keys=sort)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_C3D_path', type=str, default='')

    # ------------------------------------------------------
    # ---------- Optimization options ----------------------
    # ------------------------------------------------------
    parser.add_argument('--lr', type=float, default=0.04)  #lr从0.01调整到0.08（0.08部分效果不错），再调整到0.04
    parser.add_argument('--lr2', type=float, default=0.002, help='initial learning rate')  #lr2 从0.0005调整到 0.002（部分效果不错），再调整到0.001

    parser.add_argument('--lamda_act', type=float, default=0.5) #dropout从0.5--0.4

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--nItersLR', type=int, default=10,
                        help='Number of epochs before reducing the learning rate in a factor of 10')
    parser.add_argument('--epochs', type=int, default=80)

    parser.add_argument('--init_epoch', type=int, default=1, help='useful on restarts and train')
    parser.add_argument('--resume', type=bool, default=False, help='useful on restarts')
    parser.add_argument('--version', type=str, default='', help='time_str')
    parser.add_argument('--checkpoint_path', type=str, default='', help='resume model')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)


    # ------------------------------------------------------
    # ---------- Path options ------------------------------
    # ------------------------------------------------------
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--gpu', type=int, default=0, required=False, help='gpu id')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test', 'inference'])
    parser.add_argument('--data_path', type=str, default='./data/data_pre',
                        help='Path to preprocessed data npy files/ csv files')
    parser.add_argument('--img_path', type=str, default='./data/',
                        help='Path to preprocessed data npy files/ csv files')
    parser.add_argument('--experiment_path', type=str, default='../PainRecognition',
                        help='Path to save experiment files (results, models, logs)')
    parser.add_argument('--model_dir_name', type=str, default='checkpoints',
                        help='Name of the directory to save models')
    parser.add_argument('--result_dir_name', type=str, default='results',
                        help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--log_dir_name', type=str, default='logs',
                        help='Name of the directory to save logs (train, val)')
    # ------------------------------------------------------
    # ---------- data options ------------------------------
    # ------------------------------------------------------
    parser.add_argument('--video_size', type=tuple, default=(112, 112), help='used for transform')
    parser.add_argument('--image_size', type=tuple, default=(112, 112), help='used for transform')

    parser.add_argument('--person_id', type=int, default=0)


    # Generate args
    args = parser.parse_args()
    return args

opt = parse_args()

if __name__ == '__main__':
    save()
