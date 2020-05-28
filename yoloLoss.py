import torch
from torch.nn import *
from torch.autograd import Variable


class YoloLoss(Module):
    def __init__(self, num_class=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class

	def compute_iou(self, box1, box2):
		pass

    def compute_coord_loss(self, box1, box2):
        pass

    def compute_obj_confidenc_loss(self, cf1, cf2):
        pass

    def compute_noobj_confidenc_loss(self, cf1, cf2):
        pass

    def compute_class_loss(self, clas1, clas2):
        pass

    def forward(self, pred, target):
        return self.lambda_coord * self.compute_coord_loss(
        ) + self.compute_obj_confidenc_loss(
        ) + self.lambda_noobj * self.compute_noobj_confidenc_loss(
        ) + self.compute_class_loss()
