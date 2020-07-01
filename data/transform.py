import numpy as np
import math
from data.augmentions import ConvertColor, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans, ToTensor, Compose, ConvertFromInts


def build_transfrom(split, img_size):
    if split == 'train':
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand([123, 117, 104]),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(img_size),
            SubtractMeans([123, 117, 104]),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(img_size),
            SubtractMeans([123, 117, 104]),
            ToTensor()
        ]
    return Compose(transform)


def build_target_transform():
    return TargetTransoform()


class TargetTransoform(object):
    def __init__(self, target_shape=(7, 7, 30), class_nums=20, cell_nums=7):
        self.target_shape = target_shape
        self.class_nums = class_nums
        self.cell_nums = cell_nums

    def __call__(self, image, boxes,labels):
        """
            labels = [1,2,3,4]
            boxes = [0.2 0.3 0.4 0.8]
            return [self.S,self.S,self.B*5+self.C]
            """
        # 生成预测目标和预测分类
        np_target = np.zeros(self.target_shape)
        np_class = np.zeros((len(boxes), self.class_nums))
        for i in range(len(labels)):
            np_class[i][labels[i]] = 1
        step = 1 / self.cell_nums
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
            np_target[by][bx][:4] = box
            np_target[by][bx][4] = 1
            np_target[by][bx][5:9] = box
            np_target[by][bx][9] = 1
            np_target[by][bx][10:] = label
        return image, np_target
