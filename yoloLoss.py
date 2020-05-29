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
        # x, y, w, h = x.item(), y.item(), w.item(), h.item()
        a, b, c, d = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        return [max(a, 0), max(b, 0), min(c, 1), min(d, 1)]

    def compute_coord_loss(self, pred, target):
        mask = target[:, :, :, 8] > 0
        target_cells = target[mask][:, :8]
        pred_cells = pred[mask][:, :8]
        xy_loss = F.mse_loss(
            target_cells[:, :2], pred_cells[:, :2], reduction="sum"
        ) + F.mse_loss(target_cells[:, 4:6], pred_cells[:, 4:6], reduction="sum")

        wh_loss = F.mse_loss(
            torch.sqrt(target_cells[:, 2:4]),
            torch.sqrt(pred_cells[:, 6:8]),
            reduce="sum",
        ) + F.mse_loss(
            torch.sqrt(target_cells[:, 2:4]),
            torch.sqrt(pred_cells[:, 6:8]),
            reduce="sum",
        )
        return xy_loss + wh_loss

    def compute_iou(self, box1, box2):
        box1 = self.conver_box(box1)
        box2 = self.conver_box(box2)
        a, b, c, d = box1
        q, w, e, r = box2
        # 获取相交
        minx, miny = max(a, q), max(b, w)
        maxx, maxy = min(c, e), min(d, r)
        inter = (maxy - miny) * (maxx - minx)
        union = (e - q) * (r - q) + (d - b) * (c - a) - inter
        return inter / (union + 1e-5) if inter > 0 else 0

    def confidenc_loss(self, pred, target):
        mask = target[:, :, :, 8] > 0
        target_box = target[mask][:, :4]
        pred_box = pred[mask][:, :8]
		ious = []
		for i in range(len(target_box)):
			box =  target_box[i]
			pbox1,pbox2 = pred_box[i][:4],pred_box[i][4:8]
			iou1,iou2 = self.compute_iou(box,pbox1),self.compute_iou(box,pbox2)
			ious.apppend(max(iou1,iou2))
		noobj_mask = target[:,:,:,8]=0
		noobj_cell = pred[noobj_mask][:]
		iou_loss = 0
		for ele in noobj_cell:

			# if iou1>iou2:
			# 	pred[mask][i][8]=iou1
			# 	pred[mask][i][9]=0
			# 	ious.append()
			# else:
			# 	pred[mask][i][8]=0
			# 	pred[mask][i][9]=iou2


    def compute_obj_confidenc_loss(self, cf1, cf2):
        pass

    def compute_noobj_confidenc_loss(self, cf1, cf2):
        pass

    def compute_class_loss(self, pred, target):
        mask = target[:, :, :, 8] > 0
        pred_cells = torch.softmax(pred[mask][:, 10:], dim=1)
        target_cells = target[mask][:, 10:]
        return F.mse_loss(pred_cells, target_cells.float(), reduction="sum")

    def forward(self, pred, target):
        batch_size = pred.size(0)
        coord_loss = self.lambda_coord * self.compute_coord_loss(pred, target)
        class_loss = self.compute_class_loss(pred, target)
        self.compute_iou(pred, target)
        return (coord_loss + class_loss) / float(batch_size)


if __name__ == "__main__":
    from model import yolo
    from dataset import VOCDataset
    from torch.utils.data import DataLoader
    from torch.optim import SGD, Adam

    # train_loader = DataLoader(VOCDataset(mode='train'), batch_size=8)

    loss = YoloLoss()
    box1 = torch.Tensor([1, 2, 3, 4])
    box2 = torch.Tensor([2, 2, 4, 5])
    print(loss.compute_iou(box1, box2))
    # opt = SGD(model.parameters(), lr=0.0002, momentum=0.90, weight_decay=5e-3)
    loss_list = []
    import numpy as np

    i = 0
    from tqdm import tqdm

    model = yolo()
    img = torch.rand((2, 3, 448, 448))
    target = torch.zeros((2, 7, 7, 30))
    img, target = Variable(img), Variable(target)
    pred = model(img)
    ls = loss(pred, target)
    loss_list.append(ls.item())
    opt.zero_grad()
    ls.backward()
    opt.step()
