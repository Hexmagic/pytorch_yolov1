import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *
from shapely.geometry import Polygon
import numpy as np


class YoloLoss(Module):
    def __init__(self, num_class=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class

    def conver_box(self, box, index):
        x, y, w, h = box
        i, j = index
        step = 1 / self.S
        x = (x + j) * step
        y = (y + i) * step
        # x, y, w, h = x.item(), y.item(), w.item(), h.item()
        a, b, c, d = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        return [
            max(a.item(), 0),
            max(b.item(), 0),
            min(c.item(), 1),
            min(d.item(), 1)
        ]

    def compute_iou(self, box1, box2, index):
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        a, b, c, d = box1
        q, w, e, r = box2
        # 获取相交
        minx, miny = max(a, q), max(b, w)
        maxx, maxy = min(c, e), min(d, r)
        inter = (maxy - miny) * (maxx - minx)
        union = (e - q) * (r - w) + (d - b) * (c - a) - inter
        if inter > 0:
            iou = (inter / (union + 1e-5))
            # if iou > 1:
            # 	print('1')
            return iou
        return 0

    def forward(self, pred, target):
        batch_size = pred.size(0)
        mask = target[:, :, :, 4] > 0
        target_cell = target[mask]
        pred_cell = pred[mask]
        obj_loss = 0
        arry = mask.cpu().numpy()
        indexs = np.argwhere(arry == True)
        for i in range(len(target_cell)):
            box = target_cell[i][:4]
            index = indexs[i][1:]
            pbox1, pbox2 = pred_cell[i][:4], pred_cell[i][5:9]
            iou1, iou2 = self.compute_iou(box, pbox1, index), self.compute_iou(
                box, pbox2, index)
            if iou1 > iou2:
                target_cell[i][4] = iou1
            else:
                target_cell[i][4] = iou2
                pred_cell[i][:4] = pbox2
                pred_cell[i][4] = pred_cell[i][9]

        noobj_mask = target[:, :, :, 4] == 0
        noobj_pred = pred[noobj_mask][:, :10].contiguous().view(-1, 5)
        noobj_target = target[noobj_mask][:, :10].contiguous().view(-1, 5)

        noobj_loss = F.mse_loss(noobj_target[:, 4],
                                noobj_pred[:, 4],
                                reduction='sum')
        obj_loss = F.mse_loss(pred_cell[:, 4],
                              target_cell[:, 4],
                              reduction='sum')
        xy_loss = F.mse_loss(pred_cell[:, :2],
                             target_cell[:, :2],
                             reduction='sum')
        wh_loss = F.mse_loss(pred_cell[:, 2:4],
                             target_cell[:, 2:4],
                             reduction='sum')

        class_loss = F.mse_loss(pred_cell[:, 10:],
                                target_cell[:, 10:],
                                reduction='sum')
        loss = (obj_loss + self.lambda_noobj * noobj_loss +
                self.lambda_coord * xy_loss + self.lambda_coord * wh_loss +
                class_loss)
        return loss / batch_size


