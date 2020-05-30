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
        x = x * step + i * step
        y = y * step + j * step
        # x, y, w, h = x.item(), y.item(), w.item(), h.item()
        a, b, c, d = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        return [max(a, 0), max(b, 0), min(c, 1), min(d, 1)]

    # def coord_loss(self, pred, target):
    # 	mask = target[:, :, :, 4] > 0
    # 	target_cells = target[mask][:, :10].contiguous().view((-1,5))
    # 	pred_cells = pred[mask][:, :10].contiguous().view((-1,5))
    # 	xy_loss = F.mse_loss(
    # 		target_cells[:, :2], pred_cells[:, :2],
    # 		reduction="sum")

    # 	wh_loss = F.mse_loss(
    # 		torch.sqrt(target_cells[:, 2:4]),
    # 		torch.sqrt(pred_cells[:, 2:4]),
    # 		reduce="sum",
    # 	)
    # 	return xy_loss + wh_loss

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
            iou = (inter / (union + 1e-5)).item()
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


if __name__ == "__main__":
    from model import yolo
    from dataset import VOCDataset
    from torch.utils.data import DataLoader
    from torch.optim import SGD, Adam

    train_loader = DataLoader(VOCDataset(mode='train'), batch_size=16)

    loss = YoloLoss().cuda()

    # box1 = torch.Tensor([1, 2, 3, 4])
    # box2 = torch.Tensor([2, 2, 4, 5])
    #print(loss.compute_iou(box1, box2))

    loss_list = []
    import numpy as np

    from tqdm import tqdm

    model = yolo().cuda()
    opt = SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)
    #opt = Adam(model.parameters(),lr=0.0001)
    bar = tqdm(train_loader)
    i = 0
    for epoch in range(3):
        for img, target, line in bar:
            img, target = Variable(img).cuda(), Variable(target).cuda()
            pred = model(img)
            #print(line)
            ls = loss(pred, target.float())
            loss_list.append(ls.item())
            i += 1
            if i % 10 == 0:
                bar.set_description_str(f"loss {np.mean(loss_list)}")
            opt.zero_grad()
            ls.backward()
            opt.step()
