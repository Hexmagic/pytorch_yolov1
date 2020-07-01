from torch import nn
from model.vgg import build_vgg
import torch

class YoLo(nn.Module):
    def __init__(self, features, num_classes=20):
        super(YoLo, self).__init__()
        self.features = features
        self.classify = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                      nn.Dropout(), nn.LeakyReLU(inplace=True),
                                      nn.Linear(4096, 1470))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        x = torch.sigmoid(x)
        return x.view(-1, 7, 7, 30)


def build_yolo():
    vgg = build_vgg()
    net = YoLo(vgg.features)
    return net
