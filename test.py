import torch
from torch import nn
from torch.utils.data import DataLoader

from Data.medmnist_loader import MedMNISTDataset
from Models.SRResNet import SRResNet
from Train.train_one_epoch import eval_epoch

l = 28
scale = int(224 / 28)
print(scale)

test_dataset = MedMNISTDataset(mode='test', low_size=l)

# Define dataloaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = 'cuda'
model = SRResNet(scaling_factor=scale)
weights = torch.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/PhDResearch/Code/Federated-SuperResolution/Saved/pneumonia/best_model_8_resnet.pth')
model.load_state_dict(weights, strict=False)
model = model.to(device)
criterion = nn.L1Loss().to(device)

test_loss, test_psnr = eval_epoch(data_loader=test_loader, model=model, device=device, criterion=criterion)
print('Test PSNR: ', test_psnr)
