import numpy as np
import os
import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.00015,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--date', type=str, default='09-25-24', help='Set date for experiments')
    parser.add_argument('--gpu_ids', default='0,1', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--data_path', type=str, default='/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/')
    args = parser.parse_args()
    return args

def main():
    # Argparse
    args = args_parser()