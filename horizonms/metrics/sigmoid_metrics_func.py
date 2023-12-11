__all__ = ("sigmoid_accuracy")


def sigmoid_accuracy(ytrue, ypred, threshold=0.5, return_average=True):
    r"""Accuracy for sigmoid output.

    Args:
        ytrue (Tensor): ground truth.
        ypred (Tensor): prediction.
        threshold (float): threshold of the sigmoid output. Default is 0.5.
        return_average (bool): whether to average among samples.
    """
    assert ytrue.shape == ypred.shape
    assert ytrue.dim() == 2

    acc = (ytrue * (ypred>threshold)).sum(dim=0) / ytrue.shape[0]
    if return_average:
        return acc.mean()
    else:
        return acc