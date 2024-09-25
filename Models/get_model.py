import torchvision
from torch import nn
import torch
from Models.unet import UNet


def get_model(args, architecture):
    if architecture == 'unet':
        model = UNet(n_input_channels=1, n_output_channels=6)
    return