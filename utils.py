import argparse
import numpy as np
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--network', type=str, default='UNet')
    parser.add_argument('--loss', type=str, default='CrossEntropy')

    parser.add_argument('--bz', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)

    return parser.parse_args()


def evaluate(pred, label, alpha=0.5):
    n_class = len(set(label.flatten().tolist()))
    distribution = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            distribution[i, j] = np.sum((pred == i) * (label == j))
    # print(distribution)
    distribution /= distribution.sum()

    ### Pixel Acc
    pixelacc = np.mean(pred == label)

    ### mIoU
    IOUs = []
    for pos in range(n_class):
        A1 = np.sum(pos == pred)
        A2 = np.sum(pos == label)
        intersection = np.sum((pos == label) * (pos == pred))
        IOUs.append(intersection / (A1 + A2 - intersection))
    mIoU = np.mean(IOUs)

    ### Vrand
    vrand = (distribution ** 2).sum() / (
            alpha * (distribution.sum(1) ** 2).sum() + (1 - alpha) * (distribution.sum(0) ** 2).sum())

    ### Vinfo
    eps = 1e-7
    mutual_info = (distribution * np.log(distribution + eps)).sum() - \
                  (distribution.sum(0) * np.log(distribution.sum(0) + eps)).sum() - \
                  (distribution.sum(1) * np.log(distribution.sum(1) + eps)).sum()
    H1 = - (distribution.sum(1) * np.log(distribution.sum(1) + eps)).sum()
    H2 = - (distribution.sum(0) * np.log(distribution.sum(0) + eps)).sum()
    vinfo = mutual_info / ((1 - alpha) * H1 + alpha * H2)

    return OrderedDict([
        ('PixelAcc', pixelacc),
        ('mIoU', mIoU),
        ('Vrand', vrand),
        ('Vinfo', vinfo)
    ])
