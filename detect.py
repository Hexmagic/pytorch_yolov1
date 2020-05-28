import torch
from dataset import VOCDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


def test():
    data = VOCDataset(mode='test')
    test_loader = DataLoader(data,
                             shuffle=True,
                             num_workers=2,
                             drop_last=True,
                             batch_size=2)
    model = torch.load('weights/119_net.pk').cuda()
    for batch in test_loader:
        import pdb
        pdb.set_trace()
        img, target = batch
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        pred = model(img)
        print(pred)


if __name__ == '__main__':
    test()