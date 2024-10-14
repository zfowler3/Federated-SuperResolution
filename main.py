import copy

import numpy as np
import os
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Data.dataloader import InlineLoader
from Models.segnet import FaciesSegNet
from Models.unet_seg import UNet
from Train.train_one_epoch import train_epoch, eval_epoch
from Utils.get_seismic_data import get_dataset_seismic
from Utils.local_update import LocalUpdate


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
    args.num_classes = 6

    # Get models for each client
    model = UNet(n_channels=1, n_classes=6)
    # Transforms for test
    data_transforms_test = transforms.Compose([transforms.ToTensor()])
    # Get dataset and groups
    train_norm, test_norm, user_groups, test2_norm = get_dataset_seismic(args, transform=None)

    args.user_groups = user_groups

    # Set up Global Testing
    test_labels = np.load(args.data_path + 'test_once/test2_labels.npy')
    testlab2 = np.load(args.data_path + 'test_once/test1_labels.npy')
    # Set up dataloaders
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


    device = 'cuda'
    model = model.to(device)
    # Set up all local models in dictionary
    local_models = {
        i: {"model": None} for i in range(C)
    }
    for j in range(len(local_models)):
        local_models[j]["model"] = copy.deepcopy(model)

    # Train all clients
    for c in range(C):
        current_client = LocalUpdate(args, client_idx=c)
        loaded_model = copy.deepcopy(local_models[c]["model"])
        updated_weights = current_client.update_weights(model=copy.deepcopy(loaded_model))
        # Load in updated weights and save off model for particular client
        loaded_model.load_state_dict(updated_weights)
        local_models[c]["model"] = copy.deepcopy(loaded_model)

    # Testing
    print('Testing')
    # For each local model, pass in test images. Save off outputs, scores, etc. for final aggregation
    # Dictionary for saving off outputs for test set 2
    local_preds_2 = {
        i: {"pred": None} for i in range(C)
    }
    local_psnr_2 = {
        i: {"psnr": []} for i in range(C)
    }
    criterion = nn.CrossEntropyLoss().to(device)
    # Iterate through all clients
    for c in range(C):
        client_model = copy.deepcopy(local_models[c]["model"]) # Load current local model
        test_loss, cur_preds, cur_psnr = eval_epoch(data_loader=test_loader, model=client_model, criterion=criterion,
                                                    device=device, save_file=test_labels)
        local_preds_2[c]["pred"] = cur_preds
        local_psnr_2[c]["psnr"] = cur_psnr

    # Compare and aggregate


    #np.save('/home/zoe/ex_preds.npy', preds)

# start main
if __name__ == "__main__":
    main()