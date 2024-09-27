import numpy as np
import torch
from torch.utils.data import DataLoader
from Data.medmnist_loader import MedMNISTDataset
from Models.simplified_unet import Unet_Modified

#######
# PneumoniaMNIST
######################################################################
# Define datasets
l = 28
scale = int(224 / 28)

train_dataset = MedMNISTDataset(mode='train', low_size=l)
valid_dataset = MedMNISTDataset(mode='val', low_size=l)
test_dataset = MedMNISTDataset(mode='test', low_size=l)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

#####################################################################
# Define training params and model
epochs = 60
lr = 0.001
device = 'cuda'
model = Unet_Modified(low_size=l, scale=scale)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)