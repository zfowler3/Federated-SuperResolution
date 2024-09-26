import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Data.dataloader import InlineLoader
from Models.segnet import FaciesSegNet

data_transforms = transforms.Compose([
    transforms.ToTensor()
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5)
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
# Create datasets and loaders for train and validations
train_dataset = InlineLoader(seismic_cube=train, label_cube=train_labels, inline_inds=list(np.arange(0, train.shape[1])), train_status=True, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
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
model = FaciesSegNet(n_class=6).to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75)
criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = nn.MSELoss().to(device)

# train
epoch_loss = []
for ep in range(epochs):
    batch_loss = []
    for batch_idx, (images, labels, idx) in enumerate(train_loader):
        images, labels = images.to(device).type(torch.float), labels.to(device).type(torch.long)
        if batch_idx == 0:
            print('Train images: ', images.shape)
            testing = images[0]
            plt.imshow(testing[:, 0, :].T)
            plt.savefig('/home/zoe/ground_truth.png')
        output, reconstruct = model(images)  #
        optimizer.zero_grad()
        seg_loss = criterion1(output, labels)
        reconstruction_loss = criterion2(reconstruct, images)
        loss = seg_loss + reconstruction_loss
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    epoch_loss.append(sum(batch_loss) / len(batch_loss))
    # Validate

    print('Train loss for epoch: ', sum(batch_loss) / len(batch_loss))

# inference
print('---Training complete---')
print('Testing trained model . . .')

model.eval()
loss = 0
total = 0
test_loss = []
pred_test_vol = np.zeros(test.shape)

for batch_idx, (images, labels, idx) in enumerate(test_loader):
    images, labels = images.to(device).type(torch.float), labels.to(device).type(torch.long)
    with torch.no_grad():
        outputs = model(images)['out']
        pred_test_vol[:, batch_idx, :] = outputs.argmax(1).detach().cpu().numpy().T.squeeze()
        batch_loss = criterion1(outputs, labels)
        loss += batch_loss.item()
        test_loss.append(batch_loss.item())
total_loss = sum(test_loss)/len(test_loss)
print('Testing Loss: ', total_loss)
miou_test = jaccard_score(test_labels.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average='weighted')
miou_test_class = jaccard_score(test_labels.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average=None)
loss = 0
total = 0
test_loss = []
pred_test_vol = np.zeros(test2.shape)

for batch_idx, (images, labels, idx) in enumerate(test_loader2):
    images, labels = images.to(device).type(torch.float), labels.to(device).type(torch.long)
    with torch.no_grad():
        outputs = model(images)['out']
        pred_test_vol[:, batch_idx, :] = outputs.argmax(1).detach().cpu().numpy().T.squeeze()
        batch_loss = criterion1(outputs, labels)
        loss += batch_loss.item()
        test_loss.append(batch_loss.item())
total_loss = sum(test_loss)/len(test_loss)
print('Testing Loss: ', total_loss)
miou_test2 = jaccard_score(test_labels2.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average='weighted')
miou_test_class2 = jaccard_score(test_labels2.flatten(), pred_test_vol.flatten(), labels=list(range(6)), average=None)
print('Baseline MIOU Score for Test set 1: ', miou_test)
print('Baseline MIOU Score for Test set 2: ', miou_test2)
file_info = 'test loss, test mean iou1, test mean iou2, class miou1, class miou2, total epochs\n'
data = str(total_loss) + ', ' + str(miou_test) + ', ' + str(miou_test2) + ', ' + str(miou_test_class) + ', ' + str(miou_test_class2) + ', ' + str(epochs) + '\n'
with open("/home/zoe/Federated-Learning/Seismic-Baseline-2.txt", "w") as file:
    file.write(file_info)
    file.write(data)