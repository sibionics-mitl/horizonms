from torchvision.transforms import functional as F, InterpolationMode
from ..batch_image import BatchImage
from .segmentation_base import BaseSegmentation, get_segmentation_net
from ...builder import MODELS, build_losses_list, build_metrics_list
       

__all__ = ("Segmentation")


@MODELS.register_module()
class Segmentation(BaseSegmentation):
    r"""Class of the segmentation task for network training and testing.

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
                 batch_image=BatchImage, divisible=32,
                 batch_transforms=None):
        super(Segmentation, self).__init__(net=get_segmentation_net(net_params), 
                                final_activation=final_activation,
                                batch_image=batch_image, divisible=divisible,
                                batch_transforms=batch_transforms)
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
            for nb_level, pred in enumerate(ypred):
                if pred.shape[-1] == targets["masks"].value.shape[-1]:
                    true = targets["masks"].value
                else:
                    true = F.resize(targets["masks"].value, pred.shape[-2:], InterpolationMode.NEAREST)
                kwargs_opt = {'ypred': pred, 'ytrue': true}
                for loss_func, loss_w in zip(self.loss_funcs, self.loss_weights):
                    loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w, nb_level)
                    losses.update(loss_v)
        else:
            kwargs_opt = {'ypred': ypred, 'ytrue': targets["masks"].value}
            for loss_func, loss_w in zip(self.loss_funcs, self.loss_weights):
                loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w)
                losses.update(loss_v)
        return losses

    def calculate_metric(self, kwargs_opt, metric_func, index_head=None):
        metric_keys = metric_func.__call__.__code__.co_varnames
        metric_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in metric_keys}
        metric_v = metric_func(**metric_params)
        key_prefix = type(metric_func).__name__
        if index_head is None:
            if metric_v.dim() == 0:
                metric_v = {key_prefix: metric_v}
            else:
                metric_v = {key_prefix+'/'+str(n): v for n, v in enumerate(metric_v)}
        else:
            if metric_v.dim() == 0:
                metric_v = {key_prefix+'/'+str(index_head): metric_v}
            else:
                metric_v = {key_prefix+'/'+str(index_head)+'/'+str(n): v for n, v in enumerate(metric_v)}
        return metric_v

    def calculate_metrics(self, targets, ypred):
        metrics = {}
        if self.metric_funcs is not None:
            if isinstance(ypred, list) | isinstance(ypred, tuple):
                for nb_level, pred in enumerate(ypred):
                    if pred.shape[-1] == targets["masks"].value.shape[-1]:
                        true = targets["masks"].value
                    else:
                        true = F.resize(targets["masks"].value, pred.shape[-2:], InterpolationMode.NEAREST)
                    kwargs_opt = {'ypred': pred, 'ytrue': true}
                    for metric_func in self.metric_funcs:
                        metric_v = self.calculate_metric(kwargs_opt, metric_func, nb_level)
                        metrics.update(metric_v)
            else:
                kwargs_opt = {'ypred': ypred, 'ytrue': targets["masks"].value}
                for metric_func in self.metric_funcs:
                    metric_v = self.calculate_metric(kwargs_opt, metric_func)
                    metrics.update(metric_v)
        return metrics