from .sigmoid_metrics_func import sigmoid_accuracy
from .. import METRICS


__all__ = ("SigmoidAccuracy")


@METRICS.register_module()
class SigmoidAccuracy():
    r"""Accuracy for sigmoid output.

    Args:
        threshold (float): threshold of the sigmoid output. Default is 0.5.
        return_average (bool): whether to average among samples.
    """
    def __init__(self, threshold: float = 0.5, return_average: bool = True):
        self.threshold = threshold
        self.return_average = return_average

    def __call__(self, ytrue, ypred):
        return sigmoid_accuracy(ytrue, ypred, self.threshold, self.return_average)