import torch
from torch import nn
from torchvision.transforms import transforms
# https://github.com/quocviethere/unet-super-resolution/blob/main/model.py

class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class unetConv2d(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetConv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
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
                                nn.LeakyReLU(),
                                )
        self.conv = unetConv2d(out_channels*2, out_channels)

    def forward(self, x):
        #print(x.shape)
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

class Unet_Modified(nn.Module):
    def __init__(
            self, n_channels=1, n_classes=1, low_size=28, scale=2
    ):
        super(Unet_Modified, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize_fnc = transforms.Resize((low_size*scale, low_size*scale),
                                             antialias=True)

        # self.in_conv1 = FirstFeature(n_channels, 64)
        # self.in_conv2 = unetConv2d(64, 64)
        #
        # self.enc_1 = Encoder(64, 128)
        # self.enc_2 = Encoder(128, 256)
        # self.enc_3 = Encoder(256, 512)
        # self.enc_4 = Encoder(512, 1024)
        #
        # self.dec_1 = Decoder(1024, 512)
        # self.dec_2 = Decoder(512, 256)
        # self.dec_3 = Decoder(256, 128)
        # self.dec_4 = Decoder(128, 64)
        #
        # self.out_conv = FinalOutput(64, n_classes)

        self.in_conv1 = FirstFeature(n_channels, 8)
        self.in_conv2 = unetConv2d(8, 8)

        self.enc_1 = Encoder(8, 16)
        self.enc_2 = Encoder(16, 32)
        self.enc_3 = Encoder(32, 64)
        self.enc_4 = Encoder(64, 128)

        self.dec_1 = Decoder(128, 64)
        self.dec_2 = Decoder(64, 32)
        self.dec_3 = Decoder(32, 16)
        self.dec_4 = Decoder(16, 8)

        self.out_conv = FinalOutput(8, n_classes)

    def forward(self, x):
        x = self.resize_fnc(x)
        #print(x.shape)
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
        #print(x5.shape)
        # x = self.dec_1(x5, x4)
        # x = self.dec_2(x, x3)
        # x = self.dec_3(x, x2)
        # x = self.dec_4(x, x1)
        x = self.dec_1(x5)
        x = self.dec_2(x)
        x = self.dec_3(x)
        x = self.dec_4(x)
        x = self.out_conv(x)
        return x