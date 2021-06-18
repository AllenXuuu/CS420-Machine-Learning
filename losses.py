import torch
import torch.nn as nn


def build_loss(loss_type):
    if loss_type.lower() == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_type.lower() == 'dice':
        return DiceLoss()
    else:
        raise NotImplementedError


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, score, label):
        prob = self.softmax(score)
        onehot = nn.functional.one_hot(label, 2).permute((0, 3, 1, 2))
        intersection = (prob * onehot).sum()
        A = prob.sum()
        B = onehot.sum()
        dice = 2 * intersection / (A + B)

        return 1 - dice
