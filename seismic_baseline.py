import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Data.dataloader import InlineLoader
from Models.unet_seg import UNet
from Train.train_one_epoch import train_epoch, eval_epoch

torch.manual_seed(1)
np.random.seed(1)

data_transforms = transforms.Compose([
    transforms.ToTensor()
])
data_transforms_test = transforms.Compose([
    transforms.ToTensor()
])
# Create train and test loaders
# In this baseline case, train is all of the train numpy file
train_data = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/train/train_seismic.npy')
train = (train_data - train_data.min()) / (train_data.max() - train_data.min())
train_labels = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/train/train_labels.npy')
# Determine validation sets
valid_1_data = train[:50, :-50, :]
valid_2_data = train[:, -50:, :]
x = train[50:, :-50, :]
train = np.copy(x)
valid_1_labels = train_labels[:50, :-50, :]
valid_2_labels = train_labels[:, -50:, :]
train_labels = train_labels[50:, :-50, :]
# Dataset - train + valid
train_dataset = InlineLoader(seismic_cube=train, label_cube=train_labels, inline_inds=list(np.arange(0, train.shape[1])), train_status=True, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataset_1 = InlineLoader(seismic_cube=valid_1_data, label_cube=valid_1_labels, inline_inds=list(np.arange(0, valid_1_data.shape[1])), train_status=False, transform=data_transforms)
val_batch_size = 2
valid_loader_1 = DataLoader(valid_dataset_1, batch_size=val_batch_size, shuffle=True)
valid_dataset_2 = InlineLoader(seismic_cube=valid_2_data, label_cube=valid_2_labels, inline_inds=list(np.arange(0, valid_2_data.shape[1])), train_status=False, transform=data_transforms)
valid_loader_2 = DataLoader(valid_dataset_2, batch_size=val_batch_size, shuffle=True)
# Create test dataset and loader
test_data = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/test_once/test2_seismic.npy')
test_labels = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/test_once/test2_labels.npy')
test = (test_data - test_data.min()) / (test_data.max() - test_data.min())
test_dataset = InlineLoader(seismic_cube=test, label_cube=test_labels, inline_inds=list(np.arange(0, test.shape[1])), train_status=False, transform=data_transforms_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_data2 = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/test_once/test1_seismic.npy')
test_labels2 = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/test_once/test1_labels.npy')
test2 = (test_data2 - test_data2.min()) / (test_data2.max() - test_data2.min())
test_dataset2 = InlineLoader(seismic_cube=test2, label_cube=test_labels2, inline_inds=list(np.arange(0, test2.shape[1])), train_status=False, transform=data_transforms_test)
test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

# Params
device = 'cuda'
epochs = 60
lr = 0.001
# For SR model
model = UNet(n_channels=1, n_classes=6)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = nn.MSELoss().to(device)

# train
epoch_loss = []
best_loss = 1000000
counter = 0

for ep in range(epochs):
    loss, model = train_epoch(data_loader=train_loader, model=model, criterion=criterion1, optimizer=optimizer,
                              device=device, dataset='seismic', model_type='unet')
    print('Epoch {}/{}: Loss {}'.format(ep, epochs, loss))
    # Validation loop on both val sets
    val_loss1, _, _ = eval_epoch(data_loader=valid_loader_1, model=model, criterion=criterion1, device=device)
    val_loss2, _, _ = eval_epoch(data_loader=valid_loader_2, model=model, criterion=criterion1, device=device)
    val_loss = val_loss1 + val_loss2
    print('Validation loss: ', val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        best_model = copy.deepcopy(model)
    else:
        counter += 1
        if counter % 3 == 0:
            print('Reduced LR')
            lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        if counter >= 10:
            print('Early stop due to validation loss not decreasing')
            print('Stop at Epoch ' + str(ep))
            break

# inference
print('---Training complete---')
print('Testing best trained model . . .')
loss2, preds2, _ = eval_epoch(data_loader=test_loader2, model=copy.deepcopy(best_model), criterion=criterion1, device=device,
                              save_file=test_labels2)
loss1, preds1, _ = eval_epoch(data_loader=test_loader, model=copy.deepcopy(best_model), criterion=criterion1, device=device,
                              save_file=test_labels)

preds_folder = '/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/Results/baseline/'
miou_test1 = jaccard_score(test_labels2.flatten(), preds2.flatten(), labels=list(range(6)), average='weighted')
np.save(preds_folder + 'test_set_1.npy', preds2)
miou_test1_class = jaccard_score(test_labels2.flatten(), preds2.flatten(), labels=list(range(6)), average=None)
miou_test2 = jaccard_score(test_labels.flatten(), preds1.flatten(), labels=list(range(6)), average='weighted')
np.save(preds_folder + 'test_set_2.npy', preds1)
miou_test2_class = jaccard_score(test_labels.flatten(), preds1.flatten(), labels=list(range(6)), average=None)

print('Baseline MIOU Score for Test set 1: ', miou_test1)
print(miou_test1_class)
print('Baseline MIOU Score for Test set 2: ', miou_test2)
print(miou_test2_class)

# best_model.eval()
# loss = 0
# total = 0
# test_loss = []
# pred_test_vol = np.zeros(test.shape)
#
# for batch_idx, (images, labels, idx) in enumerate(test_loader):
#     images, labels = images.to(device).type(torch.float), labels.to(device).type(torch.long)
#     with torch.no_grad():
#         # Get super-resolution image
#         outputs_sr = best_model(images)
#         # Pass to FaultSegNet
#         #outputs = seg_model(outputs_sr)
#         pred_test_vol[:, batch_idx, :] = outputs.argmax(1).detach().cpu().numpy().T.squeeze()
#         batch_loss = criterion1(outputs, labels)
#         loss += batch_loss.item()
#         test_loss.append(batch_loss.item())
# total_loss = sum(test_loss)/len(test_loss)
# print('Testing Loss: ', total_loss)
# miou_test = jaccard_score(test_labels.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average='weighted')
# miou_test_class = jaccard_score(test_labels.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average=None)
# loss = 0
# total = 0
# test_loss = []
# pred_test_vol = np.zeros(test2.shape)
#
# for batch_idx, (images, labels, idx) in enumerate(test_loader2):
#     images, labels = images.to(device).type(torch.float), labels.to(device).type(torch.long)
#     with torch.no_grad():
#         outputs = model(images)['out']
#         pred_test_vol[:, batch_idx, :] = outputs.argmax(1).detach().cpu().numpy().T.squeeze()
#         batch_loss = criterion1(outputs, labels)
#         loss += batch_loss.item()
#         test_loss.append(batch_loss.item())
# total_loss = sum(test_loss)/len(test_loss)
# print('Testing Loss: ', total_loss)
# miou_test2 = jaccard_score(test_labels2.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average='weighted')
# miou_test_class2 = jaccard_score(test_labels2.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average=None)
# print('Baseline MIOU Score for Test set 1: ', miou_test)
# print('Baseline MIOU Score for Test set 2: ', miou_test2)
# file_info = 'test loss, test mean iou1, test mean iou2, class miou1, class miou2, total epochs\n'
# data = str(total_loss) + ', ' + str(miou_test) + ', ' + str(miou_test2) + ', ' + str(miou_test_class) + ', ' + str(miou_test_class2) + ', ' + str(epochs) + '\n'
# with open("/home/zoe/Federated-Learning/Seismic-Baseline-2.txt", "w") as file:
#     file.write(file_info)
#     file.write(data)