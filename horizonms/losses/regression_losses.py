from .losses_func import smooth_l1_loss, bbox_overlaps
from .. import LOSSES


__all__ = ["SmoothL1Loss", "RegressionIouLoss"]


@LOSSES.register_module()
class SmoothL1Loss():
    r"""Smooth L1 loss.

    Args:
        sigma (float): point where the loss changes from L2 to L1. Default: `3.0`.
        size_average (bool): if True, the average losses of all samples are calculated. Default: `True`.
    """
    def __init__(self, sigma: float = 3.0, size_average: bool = True):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.size_average = size_average
        
    def __call__(self, ytrue, ypred, weight=None):
        r"""
        Args:
            y_true (Tensor): grouth truth with shape (N, M). 
            y_pred (Tensor): network prediction with shape (N, M).
            sigma (float): point where the loss changes from L2 to L1.

        Returns:
            Tensor: loss values with shape (M,).
        """
        loss = smooth_l1_loss(ytrue, ypred, weight, sigma=self.sigma, size_average=self.size_average)
        return loss


@LOSSES.register_module()
class RegressionIouLoss():
    r"""IoU loss.

    Args:
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default: `"iou"`.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default: `1e-6`.
    """
    def __init__(self, mode: str, epsilon: float = 1e-10):
        super(RegressionIouLoss, self).__init__()
        assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
        self.mode = mode
        self.epsilon = epsilon

    def __call__(self, ypred, ytrue):
        r"""Calculate overlap between two set of bboxes.

        Args:
            ypred (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
            ytrue (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned `` is ``True``, then m and n must be equal.
            
        Returns:
            Tensor: shape (m,). 
        """
        loss = bbox_overlaps(ypred, ytrue, self.mode, True, self.epsilon)
        loss = 1 - loss.mean()
        return loss