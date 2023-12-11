import torch


__all__ = ("dice_coefficient", "iou_score")


def dice_coefficient(ytrue, ypred, smooth=1e-6):
    r"""Dice coefficient for network output.

    Args:
        ytrue (Tensor): ground truth.
        ypred (Tensor): prediction.
        epsilon (float): a small number for the stability of metric.
    """
    if ytrue.dim() == 4:
        dims = (0, 2, 3)
    elif ytrue.dim() == 2:
        dims = (0, )
    else:
        raise ValueError(f"The dimension of ytrue and ypred has to be 2 or 4, but got {ytrue.dim()}")
    dice = (2*torch.sum(ypred*ytrue, dim=dims)+smooth)/ \
            (torch.sum(ypred, dim=dims) + torch.sum(ytrue, dim=dims)+smooth)
    return dice


def iou_score(ytrue, ypred, smooth=1e-6):
    r"""Intersection over union for network output.

    Args:
        ytrue (Tensor): ground truth.
        ypred (Tensor): prediction.
        epsilon (float): a small number for the stability of metric.
    """
    if ytrue.dim() == 4:
        dims = (0, 2, 3)
    elif ytrue.dim() == 2:
        dims = (0, )
    else:
        raise ValueError(f"The dimension of ytrue and ypred has to be 2 or 4, but got {ytrue.dim()}")
    intersect = torch.sum(ypred*ytrue, dim=dims)
    union = torch.sum(ypred, dim=dims) + torch.sum(ytrue, dim=dims)
    iou = (intersect + smooth) / (union - intersect + smooth)
    return iou