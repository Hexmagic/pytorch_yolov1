import math
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from data.dataloader import make_train_dataloader, make_valid_dataloader
from model.loss import yoloLoss
from model.yolo import build_yolo
from utils.lr_scheduler import make_lr_scheduler, make_optimizer
import time


def record_loss(loss_dict, recoder):
    for key in loss_dict:
        recoder[key] += loss_dict[key].item()


def cal_loss(recoder, iter_i):
    loss_map = {}
    for key in recoder.keys():
        loss_map[key] = round(recoder[key] / iter_i, 2)
    loss_map['total_loss'] = round(sum(loss_map.values()), 2)
    return loss_map


def do_eval(model, val_loader, criterion):
    with torch.no_grad():
        recoder = {'reg_loss': 0, 'conf_loss': 0, 'cls_loss': 0}
        start = time.time()
        for iter_i, (img, target) in enumerate(val_loader):
            img, target = Variable(img).cuda(), Variable(target).cuda()
            output = model(img)
            loss_dict = criterion(output, target.float())
            record_loss(loss_dict, recoder)
            #loss = sum(x for x in loss_dict.values())
        loss_map = cal_loss(recoder, iter_i)
        end = time.time()
        eta = round(end - start, 2)
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(
            f"valid:total loss {loss_map['total_loss']} reg_loss {loss_map['reg_loss']} conf loss {loss_map['conf_loss']} cls_loss {loss_map['cls_loss']} eta: {eta} mem:{mem}"
        )


def train():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=120000)
    parser.add_argument("--start_iter", type=int, default=3)
    parser.add_argument("--n_cpu", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default='datasets')
    parser.add_argument("--pretrained_weights", type=str, help="预训练模型")
    parser.add_argument("--save_folder", type=str, default="weights")
    parser.add_argument("--img_size", type=int, default=448)
    opt = parser.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 加快训练

    if opt.pretrained_weights:
        model = build_yolo().cuda()
        state_dict = torch.load(opt.pretrained_weights)
        model.load_state_dict(state_dict)
    else:
        model = build_yolo().cuda()

    criterion = yoloLoss().cuda()
    optim = make_optimizer(model)
    lr_scheduler = make_lr_scheduler(optim)
    train_dataloader = make_train_dataloader(opt)
    val_dataloader = make_valid_dataloader(opt)
    model.train()
    start = time.time()
    recoder = {'reg_loss': 0, 'conf_loss': 0, 'cls_loss': 0}
    for iter_i, (img, target, _) in enumerate(train_dataloader, opt.start_iter):
        img, target = Variable(img).cuda(), Variable(target).cuda()
        output = model(img)
        optim.zero_grad()
        loss_dict = criterion(output, target.float())
        record_loss(loss_dict, recoder)
        loss = sum(x for x in loss_dict.values())
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if iter_i % 50 == 0:
            loss_map = cal_loss(recoder, iter_i)
            end = time.time()
            eta = round(end - start, 2)
            mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(
                f"Iter {iter_i}:total loss {loss_map['total_loss']} reg_loss {loss_map['reg_loss']} conf loss {loss_map['conf_loss']} cls_loss {loss_map['cls_loss']} eta: {eta} mem:{mem}"
            )
            start = end
        if iter_i % 5000 == 0:
            print("eval model")
            model.eval()
            do_eval(model, val_dataloader, criterion)
            model.train()
            torch.save(model.state_dict(),
                       f"{opt.save_folder}/{iter_i}_yolov1.pk")
    torch.save(model.state_dict(), f"{opt.save_folder}/{iter_i}_yolov1.pk")


if __name__ == "__main__":
    train()
