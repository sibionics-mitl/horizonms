from numpy.lib.arraysetops import isin
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from ..batch_image import BatchImage
from .anchors_retinanet import anchor_targets_bbox, BBoxCoder
from .detection_base import BaseDetection
from ...builder import MODELS, build_net, build_losses_list, build_metrics_list
import itertools


__all__ = ("Detection")


@MODELS.register_module()
class Detection(BaseDetection):
    r"""General model of the object detection task for network training and testing.

    Args:
        net_params (Dict): the configuration of the network.
        loss_params (Dict): the configuration of losses for training.
        metric_params (Dict): the configuration of the metrics for validation.
        batch_image: class used to convert a list of (input, target) into batch format used in network training and testing.
        divisible (int): it determines the size of the batched input such that it is divisible by `divisible` and larger than the size of the input.
        batch_transforms: batch transformation for network training.
        bbox_thresholds (Dict): the parameters for bounding boxes.
        bbox_stats (Dict): the statistics for bounding boxes.
    """
    def __init__(self, net_params, loss_params=[],
                 metric_params=[], batch_image=BatchImage, divisible=32,
                 batch_transforms=None, 
                 bbox_thresholds=dict(bbox_high_iou_thresh=0.5, bbox_low_iou_thresh=0.4),
                 bbox_stats=dict(bbox_stat_mean=[0,0,0,0], bbox_stat_std=[0.2,0.2,0.2,0.2])
                 ):
        if isinstance(net_params, dict):
            net = build_net(net_params)
        else:
            net = net_params
        super(Detection, self).__init__(net=net,
                                batch_image=batch_image, divisible=divisible,
                                batch_transforms=batch_transforms)
        
        if len(loss_params) > 0:
            if isinstance(list(itertools.chain.from_iterable(loss_params))[0], dict):
                self.loss_funcs, self.loss_weights = build_losses_list(loss_params)
            else:
                self.loss_funcs, self.loss_weights = loss_params
        else:
            self.loss_funcs, self.loss_weights = [], []

        self.metric_funcs = metric_params
        if len(metric_params) > 0:
            if isinstance(list(itertools.chain.from_iterable(metric_params))[0], dict):
                self.metric_funcs = build_metrics_list(metric_params)

        self.bbox_thresholds = bbox_thresholds
        self.num_classes = self.net.num_classes
        self.box_coder = BBoxCoder(mean=bbox_stats['bbox_stat_mean'],
                            std=bbox_stats['bbox_stat_std'])

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

    def calculate_losses(self, gts, preds):
        anchor_state = gts[-1].flatten()
        preds = [preds[k].flatten(0, -2) for k in range(len(preds))]
        gts = [gts[k].flatten(0, -2) for k in range(len(gts)-1)]
        losses = {}
        for index_head, (pred, true, loss_func_list, loss_w_list) in \
                    enumerate(zip(preds, gts, self.loss_funcs, self.loss_weights)):
            if index_head == 0:
                ind = anchor_state!=-1
            elif index_head == 1:
                ind = anchor_state==1
            kwargs_opt = {'ypred': pred[ind, :], 'ytrue': true[ind, :]}
            for loss_func, loss_w in zip(loss_func_list, loss_w_list):
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

    def calculate_metrics(self, gts, preds):
        anchor_state = gts[-1].flatten()
        preds = [preds[k].flatten(0, -2) for k in range(len(preds))]
        gts = [gts[k].flatten(0, -2) for k in range(len(gts)-1)]
        metrics = {}
        for index_head, (pred, true, metric_func_list) in \
                    enumerate(zip(preds, gts, self.metric_funcs)):
            if index_head == 0:
                ind = anchor_state!=-1
            elif index_head == 1:
                ind = anchor_state==1
            kwargs_opt = {'ypred': pred[ind, :], 'ytrue': true[ind, :]}
            for metric_func in metric_func_list:
                metric_v = self.calculate_metric(kwargs_opt, metric_func)
                metrics.update(metric_v) 
        return metrics 

    def forward_train(self, images, targets):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.preprocessing_input(images, targets)
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')
            
        preds, anchors = self.net(images, original_image_sizes)
        preds = self.net.pred_postprocessing_for_loss_calculation(preds)
        gts = self.net.get_gts(targets, anchors, original_image_sizes,
                self.box_coder, self.bbox_thresholds)
        losses = self.calculate_losses(gts, preds)
        return losses, preds

    @torch.no_grad()
    def test_one_batch(self, images, targets):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.preprocessing_input(images, targets)
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')

        preds, anchors = self.net(images, original_image_sizes)
        preds = self.net.pred_postprocessing_for_loss_calculation(preds)
        gts = self.net.get_gts(targets, anchors, original_image_sizes,
                self.box_coder, self.bbox_thresholds)
        losses = self.calculate_losses(gts, preds)
        metrics = self.calculate_metrics(gts, preds)

        all_boxes = self.net.postprocess_detections(preds, anchors, original_image_sizes, self.box_coder)
        # all_boxes = self.batch_image_postprocess(all_boxes, original_image_sizes, original_image_sizes)
        return losses, metrics, all_boxes

    @torch.no_grad()
    def predict_one_batch(self, images, return_dense_results=False):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.preprocessing_input(images, None)
        preds, anchors = self.net(images, original_image_sizes)

        ## calculate detections
        if return_dense_results:
            num_anchors = self.net.num_anchors
            feat_size = [[pred.shape[2],pred.shape[3]] for pred in preds[0]]
            split_size = [num_anchors*pred.shape[2]*pred.shape[3] for pred in preds[0]]
            preds = self.net.pred_postprocessing_for_loss_calculation(preds)
            pred_boxes = self.box_coder.decode(torch.stack(anchors, dim=0), preds[1])
            pred_boxes = torch.split(pred_boxes, split_size, dim=1)
            pred_boxes = [pred.reshape(pred.shape[0],sz[0],sz[1],num_anchors,pred.shape[2]) \
                          for sz,pred in zip(feat_size,pred_boxes)]
            pred_boxes = [pred.permute(0,3,4,1,2) for pred in pred_boxes]
            pred_cls = torch.split(preds[0], split_size, dim=1)
            pred_cls = [pred.reshape(pred.shape[0], num_anchors, self.num_classes, sz[0], sz[1]) \
                        for sz, pred in zip(feat_size, pred_cls)]
            # print("<<< pred_boxes", [v.shape for v in pred_boxes])
            # print("<<< pred_cls", [v.shape for v in pred_cls])
            all_boxes = [pred_cls, pred_boxes]
        else:
            preds = self.net.pred_postprocessing_for_loss_calculation(preds)
            all_boxes = self.net.postprocess_detections(preds, anchors, original_image_sizes)
            # all_boxes = self.batch_image_postprocess(all_boxes, images.image_sizes, original_image_sizes)
        return all_boxes