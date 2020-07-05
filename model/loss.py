import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *


class yoloLoss(Module):
    def __init__(self, num_class=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 7

    def conver_box(self, box, index):
        x, y, w, h = box
        i, j = index
        x = (x + j) * self.step
        y = (y + i) * self.step
        # x, y, w, h = x.item(), y.item(), w.item(), h.item()
        a, b, = x - w / 2, y - h / 2
        return [max(a.item(), 0), max(b.item(), 0), w, h]

    def compute_iou(self, box1, box2, index):
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        # 获取相交
        inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
        inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

        if inter_h <= 0 or inter_w <= 0:  # 代表相交区域面积为0
            return 0
        # 往下进行应该inter 和 union都是正值
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return (inter / union).item()

    def forward(self, pred, target):
        batch_size = pred.size(0)
        target_boxes = target[:, :, :, :10]
        pred_boxes = pred[:, :, :, :10]
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]

        obj_mask = target_boxes[:, :, :, 4] > 0 | target_boxes[:, :, :, 9] > 0
        
        mask = target[:, :, :, 4] > 0
        noobj_mask = target[:, :, :, 4] == 0
        target_cell = target[mask]
        pred_cell = pred[mask]
        obj_loss = 0
        indexs = torch.argwhere(arry == True).t()
        for i in range(len(target_cell)):
            box = target_cell[i][:4]
            index = indexs[i][1:]
            pbox1, pbox2 = pred_cell[i][:4], pred_cell[i][5:9]
            iou1, iou2 = (
                self.compute_iou(box, pbox1, index),
                self.compute_iou(box, pbox2, index),
            )
            if iou1 > iou2:
                target_cell[i][4] = iou1
                target_cell[i][9] = 0
                pred_cell[i][9] = 0
            else:
                target_cell[i][9] = iou2
                target_cell[i][4] = 0
                pred_cell[i][4] = 0

        tc = target_cell[..., :10].contiguous().view(-1, 5)
        pc = pred_cell[..., :10].contiguous().view(-1, 5)

        # noobj_mask = tc[..., 4] == 0
        noobj_pred = pred[:, :10][noobj_mask].contiguous().view(-1, 5)
        noobj_target = target[:, :10][noobj_mask].contiguous().view(-1, 5)

        mask = tc[..., 4] != 0
        tc = tc[mask]
        pc = pc[mask]

        noobj_loss = F.mse_loss(noobj_target[:, 4],
                                noobj_pred[:, 4],
                                reduction="sum")
        obj_loss = F.mse_loss(tc[:, 4], pc[:, 4], reduction="sum")
        xy_loss = F.mse_loss(tc[:, :2], pc[:, :2], reduction="sum")
        wh_loss = F.mse_loss(torch.sqrt(tc[:, 2:4]),
                             torch.sqrt(pc[:, 2:4]),
                             reduction="sum")

        class_loss = F.mse_loss(pred_cell[:, 10:],
                                target_cell[:, 10:],
                                reduction="sum")
        loss = dict(conf_loss=(obj_loss + self.lambda_noobj * noobj_loss) /
                    batch_size,
                    reg_loss=(self.lambda_coord * xy_loss +
                              self.lambda_coord * wh_loss) / batch_size,
                    cls_loss=class_loss / batch_size)
        return loss