from tqdm import tqdm
import torch
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from util.dataset import VOCDataset
from util.loss import YoloLoss
from model import yolo
from torch.autograd import Variable
import numpy as np
import math
from argparse import ArgumentParser


def update_lr(optimizer, epoch):
    lr = 0.001 - (epoch % 5 + 1) * 0.0001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train():
    parser = ArgumentParser()
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--save_folder", type=str, default="weights")
    param = parser.parse_args()
    if param.visdom:
        from visdom import Visdom

        dom = Visdom()
    train_loader = DataLoader(
        VOCDataset(mode="train"),
        batch_size=param.batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        VOCDataset(mode="val"),
        batch_size=16,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    net = yolo().cuda()
    # net = torch.load("weights/75_net.pk")
    criterion = YoloLoss().cuda()
    optim = SGD(
        params=net.parameters(),
        lr=0.001,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )
    # optim = Adam(params=net.parameters(), lr=1e-3, weight_decay=5e-4, eps=1e-4)
    t_obj_loss, t_nobj_loss, t_xy_loss, t_wh_loss, t_class_loss = [], [], [], [], []
    v_obj_loss, v_nobj_loss, v_xy_loss, v_wh_loss, v_class_loss = [], [], [], [], []
    valid_loss = []
    train_loss = []

    for epoch in range(0, 120):
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
            obj_loss, noobj_loss, xy_loss, wh_loss, class_loss = criterion(
                output, target.float()
            )
            loss = obj_loss + noobj_loss + xy_loss + wh_loss + 2 * class_loss
            loss.backward()
            train_loss.append(loss.item())
            t_obj_loss.append(obj_loss.item())
            t_nobj_loss.append(noobj_loss.item())
            t_xy_loss.append(xy_loss.item())
            t_wh_loss.append(wh_loss.item())
            t_class_loss.append(class_loss.item())
            optim.step()
            if i % 10 == 0:
                loss_list = [
                    np.mean(x)
                    for x in [
                        t_obj_loss,
                        t_nobj_loss,
                        t_xy_loss,
                        t_wh_loss,
                        t_class_loss,
                    ]
                ]
                train_bar.set_postfix_str(
                    "o:{:.2f} n:{:.2f} x:{:.2f} w:{:.2f} c:{:.2f}".format(*loss_list)
                )
                if param.visdom:
                    # train_bar.set_postfix_str(f"loss {np.mean(train_loss)}")
                    dom.line(train_loss, win="train", opts={"title": "Train loss"})
                    dom.line(t_obj_loss, win="obj", opts={"title": "obj"})
                    dom.line(t_nobj_loss, win="noobj", opts={"title": "noobj"})
                    dom.line(t_xy_loss, win="xy", opts={"title": "xy"})
                    dom.line(t_wh_loss, win="wh", opts={"title": "wh"})
                    dom.line(t_class_loss, win="class", opts={"title": "class"})
        if epoch % 5 == 0:
            torch.save(net, f"{param.save_folder}/{epoch}_net.pk")
        net.eval()
        with torch.no_grad():
            for i, ele in enumerate(val_bar):
                img, target = ele
                img, target = Variable(img).cuda(), Variable(target).cuda()
                output = net(img)
                obj_loss, noobj_loss, xy_loss, wh_loss, class_loss = criterion(
                    output, target.float()
                )
                v_obj_loss.append(obj_loss.item())
                v_nobj_loss.append(noobj_loss.item())
                v_xy_loss.append(xy_loss.item())
                v_wh_loss.append(wh_loss.item())
                v_class_loss.append(class_loss.item())
                loss = obj_loss + noobj_loss + xy_loss + wh_loss + class_loss
                valid_loss.append(loss.item())
                if i % 10 == 0:
                    loss_list = [
                        np.mean(x)
                        for x in [
                            v_obj_loss,
                            v_nobj_loss,
                            v_xy_loss,
                            v_wh_loss,
                            v_class_loss,
                        ]
                    ]
                    val_bar.set_postfix_str(
                        "o:{:.2f} n:{:.2f} x:{:.2f} w:{:.2f}c:{:.2f}".format(*loss_list)
                    )
                    if param.visdom:
                        dom.line(
                            valid_loss, win="valid_loss", opts=dict(title="Valid loss")
                        )

    torch.save(net, f"{param.save_folder}/{epoch}_net.pk")


def test():
    pass


if __name__ == "__main__":
    train()
