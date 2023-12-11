from .sigmoid_softmax_func import dice_coefficient, iou_score
from .. import METRICS


__all__ = ("DiceCoefficient", "IouScore")


@METRICS.register_module()
class DiceCoefficient():
    r"""Dice coefficient for network output.

    Args:
        epsilon (float): a small number for the stability of metric.
    """
    def __init__(self, epsilon: float = 1e-6):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon

    def __call__(self, ytrue, ypred):
        return dice_coefficient(ytrue, ypred, self.epsilon)


@METRICS.register_module()
class IouScore():
    r"""Intersection over union for network output.

    Args:
        epsilon (float): a small number for the stability of metric.
    """
    def __init__(self, epsilon: float = 1e-6):
        super(IouScore, self).__init__()
        self.epsilon = epsilon

    def __call__(self, ytrue, ypred):
        return iou_score(ytrue, ypred, self.epsilon)