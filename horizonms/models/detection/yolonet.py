import torch
# import torch.nn as nn
# import torch.nn.functional as F
from .anchors_yolo import BoxCoderYoloV1
# from .boxes import paired_box_iou
from .detection_base import BaseDetection
from ..batch_image import BatchImage
import numpy as np
# from . import get_net
from torchvision.ops import boxes as box_ops
from ... import build_net, build_loss, build_metric
from ...builder import MODELS


__all__ = ("YOLODetection")


@MODELS.register_module()
class YOLODetection(BaseDetection):
    r"""Class of the object detection task for Yolo training and testing.

    Args:
        net_params (Dict): the configuration of the network.
        loss_params (Dict): the configuration of losses for training.
        metric_params (Dict): the configuration of the metrics for validation.
        batch_transforms: batch transformation for network training.
        batch_image: class used to convert a list of (input, target) into batch format used in network training and testing.
        box_coder: encoder and decoder of anchors.
        nms_params (Dict): the parameters of NMS.
    """
    def __init__(self, net_params, loss_params=None, metric_params=None,
            batch_transforms=None, batch_image=BatchImage, box_coder=None, 
            nms_params=dict(nms_score_threshold=0.1, nms_iou_threshold=0.5,
                            detections_per_class=10)):
        if isinstance(net_params, dict):
            net = build_net(net_params)
            divisible = net_params['stride']
        else:
            net = net_params
            divisible = net.stride
        super(YOLODetection, self).__init__(net=net,
                                batch_image=batch_image, divisible=divisible,
                                batch_transforms=batch_transforms)                      
        self.num_classes = self.net.num_classes
        self.num_boxes = self.net.num_boxes
        self.stride = self.net.stride
        self.feature_shape = self.net.feature_shape
        self.nms_params = nms_params

        nb_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'# trainable parameters in network: {nb_params}')
        
        if box_coder is None:
            self.box_coder = BoxCoderYoloV1(self.stride)

        if isinstance(loss_params, dict):
            self.losses = build_loss(loss_params)
        else:
            self.losses = loss_params

        if isinstance(metric_params, dict):
            self.metrics = build_metric(metric_params)
        else:
            self.metrics = metric_params

    def calculate_losses(self, gts, preds, input_shape):
        return self.losses(gts, preds, input_shape, self.box_coder)

    def calculate_metrics(self, gts, preds, input_shape):
        return self.metrics(gts, preds, input_shape, self.box_coder)

    def forward_train(self, images, targets=None):
        images, targets = self.preprocessing_input(images, targets)
        if (self.feature_shape is None) & (self.stride is not None):
            self.feature_shape = self.net.get_feature_shape(self.stride, images.shape[-2:])
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')
        preds = self.forward(images)        
        gts = self.net.get_gts(self.feature_shape, images.shape[-2:], targets)
        losses = self.calculate_losses(gts, preds, input_shape=images.shape[-2:])
        return losses, preds  

    @torch.no_grad()
    def test_one_batch(self, images, targets):
        images, targets = self.preprocessing_input(images, targets)
        if (self.feature_shape is None) & (self.stride is not None):
            self.feature_shape = self.net.get_feature_shape(self.stride, images.shape[-2:])
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')
        preds = self.forward(images)        
        gts = self.net.get_gts(self.feature_shape, images.shape[-2:], targets)
        losses = self.calculate_losses(gts, preds, input_shape=images.shape[-2:])
        metrics = self.calculate_metrics(gts, preds, input_shape=images.shape[-2:])
        return losses, metrics, preds

    @torch.no_grad()
    def predict_one_batch(self, images):
        bs = len(images)
        original_image_sizes = [img.shape[-2:] for img in images]
        images = self.preprocessing_input(images, None)[0]
        if (self.feature_shape is None) & (self.stride is not None):
            self.feature_shape = self.net.get_feature_shape(self.stride, images.shape[-2:])
        # print(images.shape)
        # print(targets['bboxes'].value)
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')
        pred_cls, pred_conf, pred_txtytwth = self.forward(images)
        boxes = self.box_coder.decode(pred_txtytwth, images.shape[-2:])
        pred_cls_max, cls_inds = pred_cls.max(dim=1, keepdim=True)
        pred_cls_max = pred_cls_max.repeat(1,2,1,1)
        cls_inds = cls_inds.repeat(1,2,1,1)

        flags = (pred_conf > 0.1) | (pred_conf == pred_conf.max(dim=1, keepdim=True)[0])
        flags = flags.reshape(bs, -1)
        pred_cls_max = pred_cls_max.reshape(bs, -1)
        cls_inds = cls_inds.reshape(bs, -1)
        pred_conf = pred_conf.reshape(bs, -1)
        boxes = boxes.reshape(bs, -1, 4)
        scores = pred_cls_max * pred_conf
        results = self.net.nms_postprocessing(boxes, scores, cls_inds, 
                    original_image_sizes, self.nms_params, flags)
        return results
