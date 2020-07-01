import math
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from data.dataloader import make_dataloader, make_train_dataset
from model.loss import YoloLoss
from model.yolo import build_yolo
from utils.lr_scheduler import make_lr_scheduler, make_optimizer
import time


def cal_loss(loss_list):
    loss_map = defaultdict(list)
    length = len(loss_list)
    for loss_dict in loss_list:
        for key in loss_dict:
            loss_map[key].append(loss_dict[key])
    for key in loss_map.keys():
        loss_map[key] = round((sum(loss_map[key]) / length).item(), 2)
    loss_map['total_loss'] = sum(loss_map.values())
    return loss_map


def train():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=120000)
    parser.add_argument("--start_iter", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default='datasets')
    parser.add_argument("--pretrained_weights", type=str, help="预训练模型")
    parser.add_argument("--save_folder", type=str, default="weights")
    parser.add_argument("--img_size", type=int, default=448)
    opt = parser.parse_args()
    print(opt)
    torch.backends.cudnn.benchmark = True  # 加快训练

    if opt.pretrained_weights:
        model = torch.load(opt.pretrained_weights)
    else:
        model = build_yolo().cuda()

    criterion = YoloLoss().cuda()
    optim = make_optimizer(model)
    lr_scheduler = make_lr_scheduler(optim)
    dataset = make_train_dataset(opt)
    dataloader = make_dataloader(dataset, opt)
    model.train()
    loss_dict_list = []
    start = time.time()
    for iter_i, (img, target) in enumerate(dataloader, opt.start_iter):
        img, target = Variable(img).cuda(), Variable(target).cuda()
        output = model(img)
        optim.zero_grad()
        loss_dict = criterion(output, target.float())
        loss_dict_list.append(loss_dict)
        loss = sum(x for x in loss_dict.values())
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if iter_i % 10 == 0:
            loss_map = cal_loss(loss_dict_list)
            end = time.time()
            eta = round(end - start, 2)
            print(
                f"Iter {iter_i}:total loss {loss_map['total_loss']} reg_loss {loss_map['reg_loss']} conf loss {loss_map['conf_loss']} cls_loss {loss_map['cls_loss']} eta: {eta}"
            )
            loss_dict_list = []
            start = end
        if iter_i % 2000 == 0:
            torch.save(model, f"{opt.save_folder}/{iter_i}_net.pk")
    torch.save(model, f"{opt.save_folder}/{iter_i}_net.pk")


if __name__ == "__main__":
    train()
