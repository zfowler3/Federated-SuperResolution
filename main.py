import numpy as np
import os
import torch
import argparse
import segmentation_models_pytorch as smp
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from Data.dataloader import InlineLoader
from Train.train_one_epoch import train_epoch
from Utils.get_seismic_data import get_dataset_seismic


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: C")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--date', type=str, default='09-25-24', help='Set date for experiments')
    parser.add_argument('--gpu_ids', default='0,1', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--data_path', type=str,
                        default='/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/')
    parser.add_argument('--path', type=str,
                        default='/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/')
    args = parser.parse_args()
    return args

def main():
    # Argparse
    args = args_parser()
    # Set CUDA
    if args.gpu_ids:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        #torch.cuda.set_device(args.gpu_ids)
        args.cuda = True

    # init seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up number of ensembles
    C = args.num_users

    # Get dataset and groups
    train_norm, test_norm, user_groups, test2_norm = get_dataset_seismic(args)
    test_labels = np.load(args.data_path + 'test_once/test2_labels.npy')
    testlab2 = np.load(args.data_path + 'test_once/test1_labels.npy')
    # Set up dataloaders
    data_transforms_test = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = InlineLoader(test_norm, label_cube=test_labels, inline_inds=list(np.arange(0, test_norm.shape[1])),
                                train_status=False, transform=data_transforms_test)
    test2 = InlineLoader(test2_norm, testlab2, inline_inds=list(np.arange(0, test2_norm.shape[1])), train_status=False,
                         transform=data_transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loader2 = DataLoader(test2, batch_size=1, shuffle=False)

    # Define folder to save results
    results_path = args.path + 'Results/' + args.date + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_path = results_path + 'saved models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Get models for each client
    uploaded_models = {
        i: {"model": None} for i in range(C)
    }
    mapping = {0: 'deeplab', 1: 'fcn', 2: 'pan', 3: 'unet'}

    # Create models
    # Debugging: just create one instance of model
    device = 'cuda'
    model = smp.UnetPlusPlus(in_channels=1, classes=6)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for e in range(args.local_ep):
        user = user_groups[0]
        train_epoch(data_loader=user["train"], model=model, criterion=criterion, optimizer=optimizer, device=device,
                    epoch=e, dataset='seismic', model_type='unet')