from typing import List, Tuple

from .softmax_metrics_func import softmax_accuracy, softmax_accuracy_topk, \
                        softmax_cohen_kappa_score
from .softmax_metrics import SoftmaxAccuracy, SoftmaxAccuracyTopk, SoftmaxCohenKappaScore
from .sigmoid_metrics_func import sigmoid_accuracy
from .sigmoid_metrics import SigmoidAccuracy
from .sigmoid_softmax_func import dice_coefficient, iou_score
from .sigmoid_softmax_metrics import DiceCoefficient, IouScore


__all__ = ("softmax_accuracy", "softmax_accuracy_topk", "softmax_cohen_kappa_score",
           "SoftmaxAccuracy", "SoftmaxAccuracyTopk", "SoftmaxCohenKappaScore",
           "sigmoid_accuracy",
           "SigmoidAccuracy",
           "dice_coefficient", "iou_score",
           "DiceCoefficient", "IouScore")


def get_single_metric(metric: Tuple[str, dict]):
    return eval(metric[0])(**metric[1])


def get_metric_list(metric_params_list: List[Tuple[str, dict]]):
    metric_func_list = []
    for metric in metric_params_list:
        metric_func_list.append(eval(metric[0])(**metric[1]))
    return metric_func_list


def get_metrics(metric_params_list):
    metric_funcs_list = []
    for metric in metric_params_list:
        if isinstance(metric, list):
            metric_func_list = get_metric_list(metric)
        else:
            metric_func_list = get_single_metric(metric)
        metric_funcs_list.append(metric_func_list)
    return metric_funcs_list