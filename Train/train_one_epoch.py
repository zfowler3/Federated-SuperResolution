import numpy as np
from torch import nn
from torcheval.metrics.functional import peak_signal_noise_ratio
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from Utils.dice_score import dice_loss

def train_epoch(data_loader, model, criterion, optimizer, device, dataset, model_type='resnet'):
    model.train()
    batch_loss = []
    c = nn.MSELoss().to(device)
    save_dir = './examples/' + dataset + '/' + model_type + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (img, target) in enumerate(data_loader):
        img = img.to(device).type(torch.float)
        target = target.to(device).type(torch.long)
        output, recon = model(img)
        # if i == 0:
        #     print('Train img input: ', img.shape)
        #     print('Output shape: ', output.shape)
        #     print('Reconstruction shape: ', recon.shape)
            # r = recon.detach().cpu().numpy()
            # rr = r[0].squeeze()
            # print(rr.shape)
            # plt.imshow(r[0].squeeze())
            # plt.show()
            # o = pred_mask.detach().cpu().numpy().T.squeeze()
            # og = target.detach().cpu().numpy()
            # if epoch % 10 == 0:
            #     save_dir += 'epoch_' + str(epoch)
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     plt.imshow(og[0].squeeze(), cmap='gray')
            #     plt.savefig(save_dir + '/target.png')
            #     plt.clf()
            #     plt.imshow(o[0].squeeze(), cmap='gray')
            #     plt.savefig(save_dir + '/outputted.png')
            #     plt.clf()
            #     g = img.detach().cpu().numpy()
            #     plt.imshow(g[0].squeeze(), cmap='gray')
            #     plt.savefig(save_dir + '/input.png')
            #     plt.clf()
        optimizer.zero_grad()
        loss = criterion(output, target)
        reconstruction_loss = c(recon, img)
        loss += reconstruction_loss
        # loss += dice_loss(
        #     F.softmax(output, dim=1).float(),
        #     F.one_hot(target, num_classes=6).permute(0, 3, 1, 2).float(),
        #     multiclass=True
        # )
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    epoch_loss = sum(batch_loss) / len(batch_loss)
    #print('Train loss for current epoch: ', epoch_loss)
    return epoch_loss, model

def eval_epoch(data_loader, model, criterion, device, save_file=None):
    model.eval()
    count = 0
    #total_psnr = 0
    loss = []
    c = nn.MSELoss().to(device)
    psnr = []

    if save_file is not None:
        prediction = np.zeros(save_file.shape)
    else:
        prediction = np.array([])

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device).type(torch.float)
            target = target.to(device).type(torch.long)
            output, recon = model(image)
            # if i == 0:
            #     print('test output size : ', output.shape)
            if save_file is not None:
                prediction[:, i, :] = output.argmax(1).detach().cpu().numpy().T.squeeze()
            loss_ = criterion(output, target)
            reconstruction_loss = c(recon, image)
            loss_ += reconstruction_loss
            loss.append(loss_.item())
            count += 1
            cur_psnr = peak_signal_noise_ratio(recon, image).item()
            psnr.append(cur_psnr)

    #epoch_psnr = total_psnr / count
    epoch_loss = sum(loss) / len(loss)
    return epoch_loss, prediction, psnr

def eval_epoch_save(data_loader, model, criterion, device):
    model.eval()
    count = 0
    total_psnr = 0
    loss = []
    saved_outputs = np.zeros(shape=(len(data_loader), 224, 224))

    with torch.no_grad():
        for i, (gt_image, input) in enumerate(data_loader):
            gt_image = gt_image.to(device)
            input = input.to(device)
            output = model(input)
            image_final = output.detach().cpu().numpy().squeeze()
            saved_outputs[i] = image_final
            loss_ = criterion(output, gt_image)
            loss.append(loss_.item())
            count += 1
            total_psnr += peak_signal_noise_ratio(output, gt_image)

    epoch_psnr = total_psnr / count
    epoch_loss = sum(loss) / len(loss)
    return epoch_loss, epoch_psnr, saved_outputs