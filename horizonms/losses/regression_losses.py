from .losses_func import smooth_l1_loss, bbox_overlaps
from .. import LOSSES


__all__ = ("SmoothL1Loss", "RegressionIouLoss")


@LOSSES.register_module()
class SmoothL1Loss():
    r"""Smooth L1 loss.

    Args:
        sigma (float): the threshold at which to change between L1 and L2 loss. The value must be non-negative. Default: 3.0
        size_average (bool): whether to average among samples. 
    """
    def __init__(self, sigma: float = 3.0, size_average: bool = True):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.size_average = size_average
        
    def __call__(self, ytrue, ypred, weight=None):
        loss = smooth_l1_loss(ytrue, ypred, weight, sigma=self.sigma, size_average=self.size_average)
        return loss


@LOSSES.register_module()
class RegressionIouLoss():
    r"""Iou loss.

    Args:
        mode (str): specifies which iou loss is calculated. It can be `'iou'`, `'iof'`, or `'giou'`.
        epsilon (float): a small number for the stability of loss.
    """
    def __init__(self, mode: str, epsilon: float = 1e-10):
        super(RegressionIouLoss, self).__init__()
        assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
        self.mode = mode
        self.epsilon = epsilon

    def __call__(self, ypred, ytrue):
        loss = bbox_overlaps(ypred, ytrue, self.mode, True, self.epsilon)
        loss = 1 - loss.mean()
        return loss