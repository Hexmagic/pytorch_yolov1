import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *
from shapely.geometry import Polygon


class YoloLoss(Module):
    def __init__(self, num_class=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class

    def conver_box(self, box):
        x, y, w, h = box
        x, y, w, h = x.item(), y.item(), w.item(), h.item()
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(
                N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(
                N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(
                N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(
                N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt  # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]
                                               )  # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]
                                               )  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(
            inter)  # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(
            inter)  # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter  # [N, M, 2]
        iou = inter / union  # [N, M, 2]

        return iou

    def compute_coord_loss(self, pred, target):
        mask = target[:, :, :, 8] > 0
        target_cells = target[mask][:, :8]
        pred_cells = pred[mask][:, :8]
        xy_loss = F.mse_loss(
            target_cells[:, :2], pred_cells[:, :2],
            reduction='sum') + F.mse_loss(
                target_cells[:, 4:6], pred_cells[:, 4:6], reduction='sum')

        wh_loss = F.mse_loss(torch.sqrt(target_cells[:, 2:4]),
                             torch.sqrt(pred_cells[:, 6:8]),
                             reduce='sum') + F.mse_loss(
                                 torch.sqrt(target_cells[:, 2:4]),
                                 torch.sqrt(pred_cells[:, 6:8]),
                                 reduce='sum')
        return xy_loss + wh_loss

    def compute_obj_confidenc_loss(self, cf1, cf2):
        pass

    def compute_noobj_confidenc_loss(self, cf1, cf2):
        pass

    def compute_class_loss(self, pred, target):
        mask = target[:, :, :, 8] > 0
        pred_cells = torch.softmax(pred[mask][:, 10:], dim=1)
        target_cells = target[mask][:, 10:]
        return F.mse_loss(pred_cells, target_cells.float(), reduction='sum')

    def forward(self, pred, target):
        batch_size = pred.size(0)
        coord_loss = self.lambda_coord * self.compute_coord_loss(pred, target)
        class_loss = self.compute_class_loss(pred, target)
        self.compute_iou(pred, target)
        return (coord_loss + class_loss) / float(batch_size)


if __name__ == '__main__':
    from model import yolo
    from dataset import VOCDataset
    from torch.utils.data import DataLoader
    from torch.optim import SGD, Adam
    train_loader = DataLoader(VOCDataset(mode='train'), batch_size=8)
    model = yolo().cuda()
    loss = YoloLoss().cuda()
    opt = SGD(model.parameters(), lr=0.0002, momentum=0.90, weight_decay=5e-3)
    loss_list = []
    import numpy as np
    i = 0
    from tqdm import tqdm
    bar = tqdm(train_loader)
    for batch in bar:
        img, target = batch
        img, target = Variable(img).cuda(), Variable(target).cuda()
        pred = model(img)
        ls = loss(pred, target)
        loss_list.append(ls.item())
        i += 1
        if i % 10 == 0:
            bar.set_postfix_str(str(np.mean(loss_list)))

        opt.zero_grad()
        ls.backward()
        opt.step()
