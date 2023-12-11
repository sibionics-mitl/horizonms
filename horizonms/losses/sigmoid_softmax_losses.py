import torch
from .. import LOSSES


__all__ = ("DiceLoss")


@LOSSES.register_module()
class DiceLoss():
    r"""Dice loss for network output.

    Args:
        epsilon (float): a small number for the stability of loss.
    """
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def __call__(self, gt_mask, pred_mask):
        dice = 2*torch.sum(pred_mask*gt_mask,axis=(0,2,3))/ \
                            (torch.sum(pred_mask,axis=(0,2,3))+\
                             torch.sum(gt_mask, axis=(0,2,3))+self.epsilon)
        loss = 1 - dice
        return loss


