from copy import deepcopy
from .utils import Registry


__all__ = ("build_net", "build_detector", "build_backbone", "build_neck",
           "build_head", "build_loss", "build_metric", "build_transforms")


MODELS = Registry('models')
NETS = Registry('nets')
BACKBONES = Registry('backbones')
NECKS = Registry('necks')
HEADS = Registry('heads')
LOSSES = Registry('losses')
METRICS = Registry('metrics')
TRANSFORMS = Registry('transforms')


def build_models(cfg):
    return MODELS.build(cfg)


def build_net(cfg):
    return NETS.build(cfg)


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_metric(cfg):
    return METRICS.build(cfg)


def build_losses_list(cfg_list):
    loss_funcs, loss_weights = [], []
    for cfg in cfg_list:
        if isinstance(cfg, list):
            loss_func_list, loss_weight_list = build_losses_list(cfg)
        else:
            cfg_cp = deepcopy(cfg)
            loss_weight_list = cfg_cp.pop('loss_weight', 1.0)
            loss_func_list = build_loss(cfg_cp)
        loss_funcs.append(loss_func_list)
        loss_weights.append(loss_weight_list)
    return loss_funcs, loss_weights


def build_metrics_list(cfg_list):
    metric_funcs = []
    for cfg in cfg_list:
        if isinstance(cfg, list):
            metric_func_list = build_metrics_list(cfg)
        else:
            metric_func_list = build_metric(cfg)
        metric_funcs.append(metric_func_list)
    return metric_funcs


def build_transforms(cfg):
    if (cfg is None) or (len(cfg) == 0):
        return None
    return TRANSFORMS.build(cfg)