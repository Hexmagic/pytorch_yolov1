from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import math
import torch
from torchvision.transforms import Compose, RandomErasing, ColorJitter, ToTensor, ToPILImage, RandomErasing


class VOCDataset(Dataset):
    def __init__(self,
                 mode='train',
                 rt_name=False,
                 B=2,
                 C=20,
                 S=7,
                 *args,
                 **kwargs):
        super(VOCDataset, self).__init__(*args, **kwargs)
        self.mode = mode
        self.rt_name = rt_name
        self.B = B
        self.C = C
        self.S = S
        self.target_shape = (self.S, self.S, self.B * 5 + self.C)
        self.labels = {}
        self.transform = None
        if mode == 'train':
            self.transform = Compose([
                ToPILImage(),
                ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                ToTensor(),
                RandomErasing(),
            ])
        else:
            self.transform = Compose([
                ToPILImage(),
                ToTensor()
            ])
        with open(f'data/{self.mode}.txt', 'r') as f:
            lines = f.readlines()
            self.lines = []
            for line in lines:
                line = line.strip()
                label_line = line.replace('images',
                                          'labels').replace('.jpg', '.txt')
                arr = np.loadtxt(label_line)
                if arr.size == 0:
                    continue
                self.lines.append(line)
                self.labels[line] = arr.reshape(-1, 5)

        self.width = 448
        self.height = 448

    def __len__(self):
        return len(self.labels.keys())

    def make_target(self, labels, boxes):
        '''
        labels = [1,2,3,4]
        boxes = [0.2 0.3 0.4 0.8]
        return [self.S,self.S,self.B*5+self.C]
        '''
        # 生成预测目标和预测分类
        np_target = np.zeros(self.target_shape)
        np_class = np.zeros((len(boxes), self.C))
        for i in range(len(labels)):
            np_class[i][labels[i]-1] = 1
        step = 1 / self.S
        for i in range(len(boxes)):
            box = boxes[i]
            label = np_class[i]
            cx, cy, w, h = box
            # 获取中心点所在的格子,3.5 实际是第四个格子，但是0为第一个，所以索引为3
            bx = math.floor(cx / step)
            by = math.floor(cy / step)
            cx = cx % step / step
            cy = cy % step / step
            box = [cx, cy, w, h]
            np_target[bx][by][:4] = box
            np_target[bx][by][4] = 1
            np_target[bx][by][5:9] = box
            np_target[bx][by][9] = 1
            np_target[bx][by][10:] = label
        return np_target

    def __getitem__(self, idx: int):
        line = self.lines[idx].strip()
        img = cv2.imread(line)
        img = cv2.resize(img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.transpose(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        labels = self.labels[line][:, 0].astype(np.uint8)
        boxes = self.labels[line][:, 1:]
        target = self.make_target(labels, boxes)
        if self.rt_name:
            return img, target, line
        return img, target


if __name__ == "__main__":
    data = VOCDataset('train')
    loader = DataLoader(data,batch_size=2)
    for ele in loader:
        print(ele[0].shape)
        break
