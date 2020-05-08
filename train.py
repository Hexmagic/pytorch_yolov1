from tqdm import tqdm
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from dataset import VOCDataset
from model import yolo
from loss import Loss
from torch.autograd import Variable
import numpy as np
import math

init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64


def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    train_loader = DataLoader(VOCDataset(mode='train'),
                              batch_size=16,
                              num_workers=4,
                              drop_last=True,
                              shuffle=True)
    net = yolo().cuda()
    #net = torch.load('weights/99_net.pk')
    criterion = Loss().cuda()
    optim = SGD(params=net.parameters(),
                lr=0.001,
                momentum=0.9,
                weight_decay=5.0e-4)
    for epoch in range(100):
        bar = tqdm(train_loader, dynamic_ncols=True)
        batch_loss = []
        bar.set_description_str(f"epoch/{epoch}")
        for i, ele in enumerate(bar):
            update_lr(optim, epoch,
                      float(i) / float(len(train_loader) - 1))
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
        if epoch % 5 == 0:
            torch.save(net, f'weights/{epoch}_net.pk')
    torch.save(net, f'weights/{epoch}_net.pk')


def test():
    pass


if __name__ == "__main__":
    train()