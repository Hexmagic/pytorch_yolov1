from torch import nn
from model.vgg import build_vgg
import torch
import math

class YoLo(nn.Module):
    def __init__(self, features, num_classes=20):
        super(YoLo, self).__init__()
        self.features = features
        self.classify = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                      nn.ELU(inplace=True),
                                      nn.Dropout(),
                                      nn.Linear(4096, 1470))

    def init_param(self):
        for m in self.classify():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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
