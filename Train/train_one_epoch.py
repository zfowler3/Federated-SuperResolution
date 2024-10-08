import numpy as np
from torcheval.metrics.functional import peak_signal_noise_ratio
import torch
import matplotlib.pyplot as plt
import os

from train_segnet import pred_test_vol


def train_epoch(data_loader, model, criterion, optimizer, device, epoch, dataset, model_type='resnet'):
    model.train()
    batch_loss = []
    save_dir = './examples/' + dataset + '/' + model_type + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (img, target) in enumerate(data_loader):
        img = img.to(device).type(torch.float)
        target = target.to(device).type(torch.long)
        output = model(img)
        if i == 0:
            print('Train img input: ', img.shape)
            print('Output shape: ', output.shape)
            o = output.detach().cpu().numpy()
            og = target.detach().cpu().numpy()
            if epoch % 10 == 0:
                save_dir += 'epoch_' + str(epoch)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.imshow(og[0].squeeze(), cmap='gray')
                plt.savefig(save_dir + '/target.png')
                plt.clf()
                plt.imshow(o[0].squeeze(), cmap='gray')
                plt.savefig(save_dir + '/outputted.png')
                plt.clf()
                g = img.detach().cpu().numpy()
                plt.imshow(g[0].squeeze(), cmap='gray')
                plt.savefig(save_dir + '/input.png')
                plt.clf()
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    epoch_loss = sum(batch_loss) / len(batch_loss)
    print('Train loss for current epoch: ', epoch_loss)
    return epoch_loss, model

def eval_epoch(data_loader, model, criterion, device, save_file=None):
    model.eval()
    count = 0
    #total_psnr = 0
    loss = []
    if save_file is not None:
        prediction = np.zeros(save_file.shape)
    else:
        prediction = np.array([])

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device).type(torch.float)
            target = target.to(device).type(torch.long)
            output = model(image)
            if save_file is not None:
                prediction[:, i, :] = output.argmax(1).detach().cpu().numpy().T.squeeze()
            loss_ = criterion(output, target)
            loss.append(loss_.item())
            count += 1
            #total_psnr += peak_signal_noise_ratio(output, gt_image)

    #epoch_psnr = total_psnr / count
    epoch_loss = sum(loss) / len(loss)
    return epoch_loss, prediction

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