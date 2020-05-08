from tqdm import tqdm
import torch
from torch.optim import *
from torch.utils.data import DataLoader
from dataset import VOCDataset
from model import yolo
from loss import Loss
from torch.autograd import Variable
import numpy as np


def train():
    train_loader = DataLoader(VOCDataset(mode='train'),
                              batch_size=16,
                              num_workers=4,
                              drop_last=True,
                              shuffle=True)
    net = yolo().cuda()
    criterion = Loss().cuda()
    optim = Adam(params=net.parameters(), lr=0.001)
    for epoch in range(100):
        bar = tqdm(train_loader, dynamic_ncols=True)
        batch_loss = []
        bar.set_description_str(f"epoch/{epoch}")
        for i, ele in enumerate(bar):
            img, target = ele
            img, target = Variable(img).cuda(), Variable(target).cuda()
            output = net(img)
            optim.zero_grad()
            loss = criterion(output, target.float())
            loss.backward()
            batch_loss.append(loss.item())
            optim.step()
            if i % 5 == 0:
                bar.set_postfix_str(f"loss {np.mean(batch_loss)}")


def test():
    pass


if __name__ == "__main__":
    train()