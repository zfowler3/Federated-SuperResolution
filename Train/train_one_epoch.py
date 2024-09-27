from torcheval.metrics.functional import peak_signal_noise_ratio
import torch

def train_epoch(data_loader, model, criterion, optimizer, device):
    model.train()
    batch_loss = []
    for i, (gt_image, input) in enumerate(data_loader):
        gt_image = gt_image.to(device)
        input = input.to(device)
        output = model(input)
        if i == 0:
            print('Train img input: ', input.shape)
            print('Output shape: ', output.shape)
            print('GT shape: ', gt_image.shape)
        optimizer.zero_grad()
        loss = criterion(output, gt_image)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    epoch_loss = sum(batch_loss) / len(batch_loss)
    print('Train loss for current epoch: ', epoch_loss)
    return epoch_loss

def eval_epoch(data_loader, model, criterion, device):
    model.eval()
    count = 0
    total_psnr = 0
    loss = []

    with torch.no_grad():
        for i, (gt_image, input) in enumerate(data_loader):
            gt_image = gt_image.to(device)
            input = input.to(device)
            output = model(input)
            loss_ = criterion(output, gt_image)
            loss.append(loss_.item())
            count += 1
            total_psnr += peak_signal_noise_ratio(output, gt_image)

    epoch_psnr = total_psnr / count
    epoch_loss = sum(loss) / len(loss)
    return epoch_loss, epoch_psnr