import os
import random

import torch
from torch_geometric.data import Data


class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type
        self.count = 0
        self.reset()

    def reset(self):
        if self.best_type == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, "epoch:%d-val_loss:%.3f-val_acc:%.3f.model" % (epoch, val_loss, val_acc))
    torch.save(model, model_path)


def load_checkpoint(model_path):
    return torch.load(model_path)


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + ".pt")
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def load_model_dict(model, ckpt):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(ckpt, map_location="cuda:0"), False)
    else:
        d = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(d, True)


def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x


def target_swap(dataset, seed=0):
    """dataset.target -> RNA sequence
    dataset.edge_attr, dataset.x, dataset.edge_index --> ligand
    """
    dset_new = []
    random.seed(seed)
    inds = random.sample(list(range(len(dataset))), len(dataset))
    for i, d in enumerate(dataset):
        dset_new.append(
            Data(target=dataset[inds[i]].target, x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y)
        )
    return dset_new


def ligand_swap(dataset, seed=0):
    """dataset.target -> RNA sequence
    dataset.edge_attr, dataset.x, dataset.edge_index --> ligand
    """
    dset_new = []
    random.seed(seed)
    inds = random.sample(list(range(len(dataset))), len(dataset))
    for i, d in enumerate(dataset):
        lig = dataset[inds[i]]
        dset_new.append(Data(target=d.target, x=lig.x, edge_index=lig.edge_index, edge_attr=lig.edge_attr, y=d.y))
    return dset_new


def target_ones(dataset, seed=0):
    """dataset.target -> RNA sequence
    dataset.edge_attr, dataset.x, dataset.edge_index --> ligand
    """
    dset_new = []
    for d in dataset:
        target = torch.zeros_like(d.target)
        target[d.target != 0] = 1
        dset_new.append(Data(target=target, x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y))
    return dset_new


def target_shuffle(dataset, seed=0):
    """dataset.target -> RNA sequence
    dataset.edge_attr, dataset.x, dataset.edge_index --> ligand
    """
    dset_new = []
    torch.manual_seed(seed)
    for d in dataset:
        target = torch.zeros_like(d.target)
        rows, cols = target.shape
        perm = torch.randperm(cols)
        shuffled_target = target[:, perm]
        dset_new.append(Data(target=shuffled_target, x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y))
    return dset_new
