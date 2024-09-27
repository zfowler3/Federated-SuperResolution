from torcheval.metrics.functional import peak_signal_noise_ratio
import torch

def train_epoch(data_loader, model, criterion, optimizer, device):
    model.train()
    batch_loss = []
    for i, (gt_image, input) in enumerate(data_loader):
        gt_image = gt_image.to(device)
        input = input.to(device)


    return