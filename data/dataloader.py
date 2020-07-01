import torch
from torch.utils.data import DataLoader

from data.transform import build_target_transform, build_transfrom
from data.voc import VOCDataset
from utils.iter_sampler import IterationBasedBatchSampler


def make_dataloader(dataset, opt):
    batch_size, start_iter, n_cpu, max_iter = opt.batch_size, opt.start_iter, opt.n_cpu, opt.max_iter
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler=sampler, batch_size=batch_size, drop_last=False)
    if max_iter is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler,
                                                   num_iterations=max_iter,
                                                   start_iter=start_iter)

    data_loader = DataLoader(dataset,
                             num_workers=n_cpu,
                             batch_sampler=batch_sampler,
                             pin_memory=True)
    return data_loader


def make_train_dataset(opt):

    dataset = VOCDataset(data_dir=opt.data_dir,
                         split='train',
                         img_size=opt.img_size,
                         transform=build_transfrom('train', opt.img_size),
                         target_transform=build_target_transform())
    return dataset
