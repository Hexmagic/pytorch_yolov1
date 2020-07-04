import torch
from torch.utils.data import DataLoader

from data.transform import build_target_transform, build_transfrom
from data.voc import VOCDataset
from utils.iter_sampler import IterationBasedBatchSampler


def make_dataloader(dataset,
                    batch_size,
                    n_cpu,
                    start_iter,
                    max_iter,
                    not_stop=True):

    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler=sampler, batch_size=batch_size, drop_last=False)
    if not_stop:
        batch_sampler = IterationBasedBatchSampler(batch_sampler,
                                                   num_iterations=max_iter,
                                                   start_iter=start_iter)

    data_loader = DataLoader(dataset,
                             num_workers=n_cpu,
                             batch_sampler=batch_sampler,
                             pin_memory=True)
    return data_loader


def make_train_dataloader(opt):

    dataset = VOCDataset(data_dir=opt.data_dir,
                         split='train',
                         img_size=opt.img_size,
                         transform=build_transfrom('train', opt.img_size),
                         target_transform=build_target_transform())
    batch_size, n_cpu, start_iter = opt.val_batch_size, opt.n_cpu, opt.start_iter

    return make_dataloader(dataset,
                           batch_size,
                           n_cpu,
                           start_iter=start_iter,
                           max_iter=opt.max_iter,
                           not_stop=True)


def make_valid_dataloader(opt):

    dataset = VOCDataset(data_dir=opt.data_dir,
                         split='val',
                         img_size=opt.img_size,
                         transform=build_transfrom('val', opt.img_size),
                         target_transform=build_target_transform())
    batch_size, n_cpu = opt.val_batch_size, opt.n_cpu
    return make_dataloader(dataset,
                           batch_size,
                           n_cpu,
                           start_iter=0,
                           max_iter=opt.max_iter,
                           not_stop=False)
