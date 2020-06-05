import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.dataset import VOCDataset

VOC_CLASS_BGR = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}


class Detector(object):
    def __init__(self):
        self.color_map = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                          [128, 0, 128], [0, 128, 128], [128, 128, 128],
                          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                          [64, 0, 128], [192, 0, 128], [64, 128, 128],
                          [192, 128, 128], [0, 64, 0], [128, 64,
                                                        0], [0, 192, 0],
                          [128, 192, 0], [0, 64, 128], [0, 64, 128]]
        self.name_map = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.test_loader = DataLoader(VOCDataset(mode='train', rt_name=True),
                                      shuffle=False,
                                      num_workers=1,
                                      drop_last=True,
                                      batch_size=2)
        self.model = torch.load('weights/75_net.pk')
        self.S = 7
        if not os.path.exists('output'):
            os.mkdir('output')

    def conver_box(self, box, index):
        x, y, w, h = box
        c, r = index
        step = 1 / self.S
        x = (x + c) * step
        y = (y + r) * step
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
        pred = pred.reshape((-1, 30))
        mask = pred[:, 4] > 0.12
        indexs = np.argwhere(mask == True).T
        #print(indexs)
        for index in indexs:
            #b, a = index
            c = index % 7
            r = index // 7
            cell = pred[index][0]  #[b]

            pred_class = torch.argmax(cell[10:])
            cls_name = self.name_map[pred_class.item()]
            color = self.color_map[pred_class.item()]
            if cell[4] > cell[9]:
                box = cell[:4]
                score = cell[4]
            else:
                box = cell[5:9]
                score = cell[9]
            # if score.item() < 0.1:
            #     continue
            rect = self.conver_box(box, [c, r])
            rect = list(map(lambda x: int(448 * x), rect))
            cv2.putText(img, '{}:{:.2f}'.format(cls_name, score.item()),
                        tuple(rect[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 40), 1)
            cv2.rectangle(img, tuple(rect[:2]), tuple(rect[2:]), tuple(color),
                          1)
        img = cv2.resize(img, (h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(f'output/{name.split("/")[-1]}')

    def run(self):
        with torch.no_grad():
            self.model.eval()
            i = 0
            for batch in tqdm(self.test_loader):
                i += 1
                img, target, name = batch
                img = Variable(img).cuda()
                target = Variable(target).cuda()
                pred = self.model(img)
                pred = pred.cpu()
                for ele in zip(name, pred):
                    self.draw_box(ele)
                if i > 100:
                    break


if __name__ == '__main__':
    dec = Detector()
    dec.run()
