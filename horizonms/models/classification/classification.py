from torch.functional import Tensor
import torch.nn.functional as F
from ..batch_image import BatchImage
from .classification_base import BaseClassification, get_classification_net
from ...builder import build_losses_list, build_metrics_list, build_transforms, MODELS


__all__ = ("Classification")


@MODELS.register_module()
class Classification(BaseClassification):
    r"""Class of the classification task for network training and testing.

    Args:
        net_params (Dict): the configuration of the network.
        final_activation ('softmax' | 'sigmoid' | None): Decide which type of operator is used to the output of `net`.
            When final_activation=None, no operator is applied.
            When final_activation='softmax', softmax operator is applied.
            When final_activation='softmax', sigmoid operator is applied.
        loss_params (Dict): the configuration of losses for training.
        metric_params (Dict): the configuration of the metrics for validation.
        batch_image: class used to convert a list of (input, target) into batch format used in network training and testing.
        divisible (int): it determines the size of the batched input such that it is divisible by `divisible` and larger than the size of the input.
        batch_transforms: batch transformation for network training.
    """
    def __init__(self, net_params, final_activation=None,
                 loss_params=None, metric_params=None,
                 batch_image=BatchImage, divisible=1,
                 batch_transforms=None):
        super(Classification, self).__init__(net=get_classification_net(net_params), 
                                final_activation=final_activation,
                                batch_image=batch_image, divisible=divisible,
                                batch_transforms=build_transforms(batch_transforms))
        self.loss_funcs, self.loss_weights = build_losses_list(loss_params)
        if metric_params is not None:
            self.metric_funcs = build_metrics_list(metric_params)
        else:
            self.metric_funcs = None

    def calculate_loss(self, kwargs_opt, loss_func, loss_w, index_head=None):
        loss_keys = loss_func.__call__.__code__.co_varnames
        loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
        loss_v = loss_func(**loss_params)*loss_w

        key_prefix = type(loss_func).__name__
        if index_head is None:
            if loss_v.dim() == 0:
                loss_v = {key_prefix: loss_v}
            else:
                loss_v = {key_prefix+'/'+str(n): v for n, v in enumerate(loss_v)}
        else:
            if loss_v.dim() == 0:
                loss_v = {key_prefix+'/'+str(index_head): loss_v}
            else:
                loss_v = {key_prefix+'/'+str(index_head)+'/'+str(n): v for n, v in enumerate(loss_v)}
        return loss_v

    def calculate_losses(self, targets, ypred):
        losses = dict()
        if isinstance(ypred, list) | isinstance(ypred, tuple):
            if isinstance(targets["labels"], list):
                labels = targets["labels"]
            else:
                labels = targets["labels"].value

            for index_head, (pred, true, loss_func_list, loss_w_list) in \
                    enumerate(zip(ypred, labels, self.loss_funcs, self.loss_weights)):
                kwargs_opt = {'ypred': pred, 'ytrue': true}
                for loss_func, loss_w in zip(loss_func_list, loss_w_list):
                    loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w, index_head)
                    losses.update(loss_v)
        else:
            if isinstance(targets["labels"], Tensor):
                kwargs_opt = {'ypred': ypred, 'ytrue': targets["labels"]}
            else:
                kwargs_opt = {'ypred': ypred, 'ytrue': targets["labels"].value}
            for loss_func, loss_w in zip(self.loss_funcs, self.loss_weights):
                loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w)
                losses.update(loss_v)
        return losses

    def calculate_metric(self, kwargs_opt, metric_func, index_head=None):
        metric_keys = metric_func.__call__.__code__.co_varnames
        metric_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in metric_keys}
        metric_v = metric_func(**metric_params)
        if index_head is None:
            metric_v = {type(metric_func).__name__: metric_v}
        else:
            metric_v = {type(metric_func).__name__+'/'+str(index_head): metric_v}
        return metric_v

    def calculate_metrics(self, targets, ypred):
        metrics = {}
        if self.metric_funcs is not None:
            if isinstance(ypred, list) | isinstance(ypred, tuple):
                if isinstance(targets["labels"], list):
                    labels = targets["labels"]
                else:
                    labels = targets["labels"].value
                
                for index_head, (pred, true, metric_func_list) in \
                    enumerate(zip(ypred, labels, self.metric_funcs)):
                    kwargs_opt = {'ypred': pred, 'ytrue': true}
                    for metric_func in metric_func_list:
                        metric_v = self.calculate_metric(kwargs_opt, metric_func, index_head)
                        metrics.update(metric_v)
            else:
                if isinstance(targets["labels"], Tensor):
                    kwargs_opt = {'ypred': ypred, 'ytrue': targets["labels"]}
                else:
                    kwargs_opt = {'ypred': ypred, 'ytrue': targets["labels"].value}
                for metric_func in self.metric_funcs:
                    metric_v = self.calculate_metric(kwargs_opt, metric_func)
                    metrics.update(metric_v)
        return metrics
