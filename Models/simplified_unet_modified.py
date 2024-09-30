import torch
from torch import nn
from Models import common
import torch.nn.functional as F
import numpy as np

# https://github.com/Mnster00/simplifiedUnetSR/blob/master/Unet/Umodel.py

def pad_to(x):
    h, w = x.shape[-2:]
    arr = 2**np.arange(0, 12)
    vals_h = arr[(arr - h) >= 0]
    pad_h = vals_h[np.argmin(vals_h)]
    vals_w = arr[(arr - w) >= 0]
    pad_w = vals_w[np.argmin(vals_w)]
    stride_h = pad_h - h
    stride_w = pad_w - w

    new_h = h + stride_h - h % stride_h
    new_w = w + stride_w - w % stride_w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)
    out = F.pad(x, pads, "reflect", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class unetConv2d(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetConv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()

        self.conv = unetConv2d(in_size, out_size)
        self.down = nn.MaxPool2d(2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)

        return outputs

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                nn.Conv2d(in_channels, out_channels*2, 1, 1, 0, bias=False),
                                nn.BatchNorm2d(out_channels*2),
                                nn.ReLU(),
                                )
        self.conv = unetConv2d(out_channels*2, out_channels)

    def forward(self, x):
        x = self.up(x)
        #print(x.shape)
        #x = torch.concat([x, skip], dim=1)
        x = self.conv(x)

        return x

class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class Unet_(nn.Module):
    def __init__(
            self, n_channels=1, n_classes=1, scale=2
    ):
        super(Unet_, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        features = [64, 128, 256, 512, 1024]
        #self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = unetConv2d(n_channels, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = Decoder(1024, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)

        self.upend = common.Upsampler(common.default_conv, scale, 64, act=False)

        resblock = [common.ResBlock(
            common.default_conv, features[0], 3, act=nn.ReLU(True), bn=True, res_scale=1
            ) for _ in range(3)]
        self.resblock = nn.Sequential(*resblock)

        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x, pads = pad_to(x)
        x1 = self.in_conv2(x)
        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
        x = self.dec_1(x5)
        x = self.dec_2(x)
        x = self.dec_3(x)
        x = self.dec_4(x)
        x = unpad(x, pads)
        x = self.upend(x)
        x = self.resblock(x)
        x = self.out_conv(x)
        return x