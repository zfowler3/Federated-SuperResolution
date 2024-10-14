import torch
from torch import nn
from Train.train_one_epoch import train_epoch

class LocalUpdate(object):
    def __init__(self, args, client_idx):
        self.args = args
        self.idx = client_idx
        self.train_loader = self.args.user_groups[client_idx]["train"]
        self.train_size = self.args.user_groups[client_idx]["datasize"]
        self.nclasses = self.args.num_classes

    def update_weights(self, model):
        device = 'cuda'
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss().to(device)

        for i in range(self.args.local_ep):
            loss, model = train_epoch(data_loader=self.train_loader, model=model, criterion=criterion, optimizer=optimizer,
                                      device=device, dataset='seismic', model_type='unet')
            print('Epoch {}/{}: Loss {}'.format(i, self.args.local_ep, loss))

        return model.state_dict()