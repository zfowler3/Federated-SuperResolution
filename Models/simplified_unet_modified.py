import torch
from torch import nn
from Models import common

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
            #nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()

        self.conv = unetConv2d(in_size, out_size)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)

        return outputs

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=False),
                                #nn.BatchNorm2d(out_channels*2),
                                #nn.ReLU(),
                                )
        self.conv = unetConv2d(out_channels*2, out_channels)

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output

class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class Unet_(nn.Module):
    def __init__(
            self, n_channels=1, n_classes=1, low_size=28, scale=2
    ):
        super(Unet_, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        features = [64, 128, 256, 512, 1024]
        features = [x // scale for x in features]

        self.in_conv1 = unetConv2d(n_channels, features[0])

        self.enc_1 = Encoder(features[0], features[1])
        self.enc_2 = Encoder(features[1], features[2])
        self.enc_3 = Encoder(features[2], features[3])
        self.enc_4 = Encoder(features[3], features[4])

        self.dec_1 = Decoder(features[4], features[3])
        self.dec_2 = Decoder(features[3], features[2])
        self.dec_3 = Decoder(features[2], features[1])
        self.dec_4 = Decoder(features[1], features[0])

        self.upend = common.Upsampler(common.default_conv, scale, features[0], act=False)

        resblock = [common.ResBlock(
            common.default_conv, features[0], 3, act=nn.ReLU(True), bn=True, res_scale=1
            ) for _ in range(3)]
        self.resblock = nn.Sequential(*resblock)

        self.out_conv = FinalOutput(features[0], n_classes)

    def forward(self, x):
        #x = self.resize_fnc(x)
        #print(x.shape)
        x1 = self.in_conv1(x)
        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)
        #print(x5.shape)
        # x = self.dec_1(x5, x4)
        # x = self.dec_2(x, x3)
        # x = self.dec_3(x, x2)
        # x = self.dec_4(x, x1)
        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)
        x = self.upend(x)
        x = self.resblock(x)
        x = self.out_conv(x)
        return x