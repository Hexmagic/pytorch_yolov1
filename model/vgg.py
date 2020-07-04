import torch
import torch.nn.functional as F
from torch import nn

from utils.modelzoo import load_state_dict_from_url

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag = True
    for v in cfg:
        s = 1
        if (v == 64 and first_flag):
            s = 2
            first_flag = False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,
                               v,
                               kernel_size=3,
                               stride=s,
                               padding=1)
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.LeakyReLU(inplace=True)
                ]
            else:
                layers += [conv2d, nn.LeakyReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_vgg():
    vgg = VGG(make_layers(cfg['D'], batch_norm=True))
    vgg.load_state_dict(
        load_state_dict_from_url(
            'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'))
    return vgg
