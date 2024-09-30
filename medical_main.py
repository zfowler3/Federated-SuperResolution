import copy
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data.medmnist_loader import MedMNISTDataset
from Models.SRResNet import SRResNet
from Models.simplified_unet import Unet_Modified
from Models.simplified_unet_modified import Unet_
from Train.train_one_epoch import train_epoch, eval_epoch

#######
# PneumoniaMNIST
######################################################################
# Define datasets
l = 28
scale = int(224 / 28)
folder = '/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/PhDResearch/Code/Federated-SuperResolution/Saved/pneumonia/'
if not os.path.exists(folder):
    os.makedirs(folder)

train_dataset = MedMNISTDataset(mode='train', low_size=l)
valid_dataset = MedMNISTDataset(mode='val', low_size=l)
test_dataset = MedMNISTDataset(mode='test', low_size=l)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#####################################################################
# Define training params and model
epochs = 60
lr = 0.001
device = 'cuda'
#model = Unet_(scale=scale)
model = SRResNet(scaling_factor=scale)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.5,0.999])
criterion = nn.L1Loss().to(device)
best_psnr = 0
count = 0

for e in range(epochs):
    print('---------------------------------')
    print('Epoch {}/{}'.format(e, epochs))
    epoch_loss = []
    # Train model
    loss = train_epoch(data_loader=train_loader, model=model, optimizer=optimizer, device=device, criterion=criterion)
    epoch_loss.append(loss)
    # Eval on validation set
    val_loss, val_psnr = eval_epoch(data_loader=valid_loader, model=model, device=device, criterion=criterion)
    print('Validation loss {}, Validation PSNR {}'.format(val_loss, val_psnr))
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        best_model = copy.deepcopy(model)
        count = 0
    else:
        count += 1
        if count == 5:
            lr /= 10
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999))
        if count >= 10:
            print('Early stopping.')
            break

#####################################################
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Training complete.')
print('Best validation PSNR: {}'.format(best_psnr))
print('Begin testing best model on test set.')

test_loss, test_psnr = eval_epoch(data_loader=test_loader, model=best_model, device=device, criterion=criterion)
print('Test set PSNR: ', test_psnr)

# Save off best model
torch.save(best_model.state_dict(), folder + 'best_model_' + str(scale) + '.pth')

