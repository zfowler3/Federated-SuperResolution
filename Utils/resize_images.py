import numpy as np
from medmnist import PneumoniaMNIST
import argparse
import os
from torchvision.transforms import transforms


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pneumonia',
                        choices=['pneumonia', 'olives', 'seismic'])

    args = parser.parse_args()
    return args

def main():
    # Argparse
    args = args_parser()
    output_path = '/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Super_Resolution/'

    if args.dataset == 'pneumonia':
        output_path = output_path + 'pneumonia/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        # use MedMNIST
        train = PneumoniaMNIST(split="train", size=28, download=True, transform=data_transform)
