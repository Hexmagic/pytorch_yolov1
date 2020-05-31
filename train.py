from tqdm import tqdm
import torch
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from util.dataset import VOCDataset
from model import yolo
from util.loss import YoloLoss
from torch.autograd import Variable
import numpy as np
import math
from visdom import Visdom


def update_lr(optimizer, epoch):
    if epoch == 5:
        lr = 0.0007
    elif epoch == 15:
        lr = 0.0006
    elif epoch == 75:
        lr = 0.0003
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    dom = Visdom()
    train_loader = DataLoader(VOCDataset(mode='train'),
                              batch_size=24,
                              num_workers=4,
                              drop_last=True,
                              shuffle=True)
    valid_loader = DataLoader(VOCDataset(mode='val'),
                              batch_size=4,
                              num_workers=4,
                              drop_last=True,
                              shuffle=True)
    net = yolo().cuda()
    criterion = YoloLoss().cuda()
    optim = SGD(params=net.parameters(),
                lr=0.001,
                weight_decay=5e-4,
                momentum=0.9)
    optim = Adam(params=net.parameters())
    train_loss = []
    valid_loss = []

    for epoch in range(136):
        train_bar = tqdm(train_loader, dynamic_ncols=True)
        val_bar = tqdm(valid_loader, dynamic_ncols=True)
        train_bar.set_description_str(f"epoch/{epoch}")
        update_lr(optim, epoch)
        net.train()
        for i, ele in enumerate(train_bar):
            img, target = ele
            img, target = Variable(img).cuda(), Variable(target).cuda()
            output = net(img)
            optim.zero_grad()
            loss = criterion(output, target.float())
            loss.backward()
            train_loss.append(loss.item())
            optim.step()
            if i % 5 == 0:
                train_bar.set_postfix_str(f"loss {np.mean(train_loss)}")
                dom.line(train_loss, win='train_loss')
        if epoch % 5 == 0:
            torch.save(net, f'weights/{epoch}_net.pk')
        net.eval()
        for i, ele in enumerate(val_bar):
            img, target = ele
            img, target = Variable(img).cuda(), Variable(target).cuda()
            output = net(img)
            loss = criterion(output, target.float())
            valid_loss.append(loss.item())
            if i % 5 == 0:
                val_bar.set_postfix_str(f"loss {np.mean(valid_loss)}")
                dom.line(valid_loss, win='valid_loss', opts=dict())

    torch.save(net, f'weights/{epoch}_net.pk')


def test():
    pass


if __name__ == "__main__":
    train()
