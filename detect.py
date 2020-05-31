import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.dataset import VOCDataset


class Detector(object):
    def __init__(self):
        self.test_loader = DataLoader(VOCDataset(mode='train', rt_name=True),
                                      shuffle=True,
                                      num_workers=1,
                                      drop_last=True,
                                      batch_size=1)
        self.model = torch.load('weights/70_net.pk')
        self.S = 7
        if not os.path.exists('output'):
            os.mkdir('output')

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

    def draw_box(self, zp):
        name, pred = zp
        img = cv2.imread(name)
        h, w, _ = img.shape
        img = cv2.resize(img, (448, 448))
        mask = (pred[:, :, 4] > 0.2) | (pred[:, :, 9] > 0.2)
        indexs = np.argwhere(mask == True).T
        print(indexs)
        for index in indexs:
            a, b = index
            cell = pred[a][b]
            #pred_class = torch.argmax(cell[10:])
            if cell[4] > cell[9]:
                box = cell[:4]
                score = cell[4]
            else:
                box = cell[5:9]
                score = cell[9]
            rect = self.conver_box(box, [a, b])
            rect = list(map(lambda x: int(448 * x), rect))
            cv2.rectangle(img, tuple(rect[:2]), tuple(rect[2:]),
                          (255, 123, 255), 1)
        img = cv2.resize(img, (h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(f'output/{name.split("/")[-1]}')

    def run(self):
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(self.test_loader):
                img, target, name = batch
                img = Variable(img).cuda()
                target = Variable(target).cuda()
                pred = self.model(img)
                pred = pred.cpu()
                for ele in zip(name, pred):
                    self.draw_box(ele)


if __name__ == '__main__':
    dec = Detector()
    dec.run()
