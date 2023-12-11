import torch
from .base import SigmoidBaseLoss
from .. import LOSSES


__all__ = ("SigmoidCrossEntropyLoss", "SigmoidFocalLoss")


@LOSSES.register_module()
class SigmoidCrossEntropyLoss(SigmoidBaseLoss):
    r"""Cross entropy loss for sigmoid output.

    Args:
        mode (str): the mode of cross entropy loss. It is 'all' or 'balance'. Default: ``'all'``.
            These two modes are different in how to get focal loss from individual samples.
            'all' returns an average for all samples.
            'balance' returns an average for all class, each class also returns an average for all of its samples.
    """
    def __init__(self, mode='all', *argv, **kwargs):
        super(SigmoidCrossEntropyLoss, self).__init__(*argv, **kwargs)
        self.mode = mode

    def calculate_loss(self, ytrue, ypred, flag=None):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        if flag is not None:
            ytrue_pos = ytrue * flag
            ytrue_neg = (1-ytrue) * flag
        else:
            ytrue_pos = ytrue
            ytrue_neg = 1 - ytrue
        loss_pos = -ytrue_pos*torch.log(ypred)
        loss_neg = -ytrue_neg*torch.log(1-ypred)

        loss_pos = torch.sum(loss_pos, dim=0)
        loss_neg = torch.sum(loss_neg, dim=0)
        nb_pos = torch.sum(ytrue_pos, dim=0)
        nb_neg = torch.sum(ytrue_neg, dim=0)

        if self.mode=='all':
            loss  = (loss_pos+loss_neg)/(nb_pos+nb_neg)
        elif self.mode=='balance':
            loss  = (loss_pos/nb_pos+loss_neg/nb_neg)/2        
        return loss


@LOSSES.register_module()
class SigmoidFocalLoss(SigmoidBaseLoss):
    r"""Focal loss for sigmoid output.

    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor to balance easy vs hard examples. Default: ``2.0``.
        cutoff (float): the threshold to determine positive and negative classes in ground truth. Default: ``0.5``.
    """
    def __init__(self, alpha=0.25, gamma=2.0, cutoff=0.5, *argv, **kwargs):
        super(SigmoidFocalLoss, self).__init__(*argv, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.cutoff = cutoff
        
    def calculate_loss(self, ytrue, ypred, flag=None):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)

        # compute the focal loss
        alpha_factor = torch.ones_like(ytrue) * self.alpha
        alpha_factor = torch.where(ytrue > self.cutoff, alpha_factor, 1-alpha_factor)
        focal_weight = torch.where(ytrue > self.cutoff, 1-ypred, ypred)
        focal_weight = alpha_factor * focal_weight ** self.gamma

        bce = -ytrue*torch.log(ypred) - (1-ytrue)*torch.log(1-ypred)
        if flag is None:
            cls_loss = focal_weight * bce
            normalizer = torch.sum(ytrue, dim=0)
        else:
            cls_loss = focal_weight * bce * flag
            normalizer = torch.sum(flag*ytrue, dim=0)

        normalizer += (normalizer < 1).long()
        loss = torch.sum(cls_loss, dim=0) / normalizer

        return loss