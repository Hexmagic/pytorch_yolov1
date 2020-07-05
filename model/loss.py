import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *


class yoloLoss(Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 7

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1], box[:, 2], box[:, 3] = [
            (box[:, 0] + i) * self.step - box[:, 2] / 2,
            (box[:, 1] + j) * self.step - box[:, 3] / 2,
            (box[:, 0] + i) * self.step + box[:, 2] / 2,
            (box[:, 1] + j) * self.step + box[:, 3] / 2
        ]
        return box

    def compute_iou(self, box1, box2, index):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M,
                                            2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M,
                                            2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M,
                                            2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M,
                                            2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred, target):
        batch_size = pred.size(0)
        target_boxes = target[:, :, :, :10].contiguous().reshape(
            (-1, 7, 7, 2, 5))
        pred_boxes = pred[:, :, :, :10].contiguous().reshape((-1, 7, 7, 2, 5))
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]
        obj_mask = (target_boxes[..., 4] > 0).byte()
        sig_mask = obj_mask[..., 1].bool()
        index = torch.where(sig_mask == True)

        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x]
            target_box = target_boxes[img_i, y, x, 0:1]
            ious = self.compute_iou(pbox[:, :4], target_box[:, :4], [x, y])
            iou, max_i = ious.max(0)
            pred_boxes[img_i, y, x, max_i, 4] = iou.item()
            pred_boxes[img_i, y, x, 1 - max_i, 4] = 0
            obj_mask[img_i, y, x, 1 - max_i] = 0
        obj_mask = obj_mask.bool()
        noobj_mask = ~obj_mask

        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")
        xy_loss = F.mse_loss(pred_boxes[obj_mask][:, :2],
                             target_boxes[obj_mask][:, :2],
                             reduction="sum")
        wh_loss = F.mse_loss(torch.sqrt(target_boxes[obj_mask][:,2:4]),
                             torch.sqrt(pred_boxes[obj_mask][:,2:4]),
                             reduction="sum")
        class_loss = F.mse_loss(pred_cls[sig_mask],
                                target_cls[sig_mask],
                                reduction="sum")

        loss = dict(conf_loss=(obj_loss + self.lambda_noobj * noobj_loss) /
                    batch_size,
                    reg_loss=(self.lambda_coord * xy_loss +
                              self.lambda_coord * wh_loss) / batch_size,
                    cls_loss=class_loss / batch_size)
        return loss