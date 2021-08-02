import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from data.augmentions import *
from sys import platform


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor')
    sep = '\\' if platform == 'win32' else '/'

    def __init__(self,
                 data_dir,
                 split,
                 transform=None,
                 target_transform=None,
                 img_size=448,
                 years=[2007, 2012],
                 keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split

        if split != 'test':
            image_sets_file = [
                os.path.join(self.data_dir, f'VOC{year}', "ImageSets", "Main",
                             "%s.txt" % self.split) for year in years
            ]
            self.ids = VOCDataset._read_image_ids(image_sets_file)
        else:
            image_sets_file = [
                os.path.join(self.data_dir, f'VOC{year}', "ImageSets", "Main",
                             "%s.txt" % self.split) for year in [2007]
            ]
            self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        self.class_dict = {
            class_name: i
            for i, class_name in enumerate(self.class_names)
        }

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
            boxes = np.clip(boxes, 0.0, 1.0)
        if self.target_transform:
            image, targets = self.target_transform(image, boxes, labels)
            return image, targets, image_id
        return image, boxes, labels

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_files):
        ids = []
        for filename in image_sets_files:
            with open(filename) as f:
                lst = filename.split(VOCDataset.sep)
                lst = lst[:-1]
                lst[2] = 'Annotations'
                for line in f:
                    lst[3] = f'{line.strip()}.xml'
                    ids.append(VOCDataset.sep.join(lst))
        return ids

    def _get_annotation(self, image_id):
        annotation_file = image_id
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cx,cy = (x1+x2)/2,(y1+y2)/2
            w,h = x2-x1,y2-y1
            boxes.append([cx, cy, w, h])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(
                int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes,
                         dtype=np.float32), np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations",
                                       "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(
            map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        lst = image_id.split(VOCDataset.sep)
        lst[2] = 'JPEGImages'
        lst[3] = lst[3].replace('.xml', '.jpg')
        image_file = VOCDataset.sep.join(lst)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
