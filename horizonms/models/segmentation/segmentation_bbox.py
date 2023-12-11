import torch
from torchvision.transforms import functional as F, InterpolationMode
from ..batch_image import BatchImage
from .segmentation_base import BaseSegmentation, get_segmentation_net
from ...builder import MODELS, build_losses_list, build_metrics_list


__all__ = ("BboxSegmentation")


@MODELS.register_module()
class BboxSegmentation(BaseSegmentation):
    r"""Class of the weakly supervised image segmentation based on bounding boxes for network training and testing.

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
    """
    def __init__(self, net_params, final_activation=None,
                 loss_params=None, metric_params=None,
                 batch_image=BatchImage, divisible=32):
        super(BboxSegmentation, self).__init__(net=get_segmentation_net(net_params), 
                                final_activation=final_activation,
                                batch_image=batch_image, divisible=divisible,
                                batch_transforms=None)
        self.loss_funcs, self.loss_weights = build_losses_list(loss_params)
        if metric_params is not None:
            self.metric_funcs = build_metrics_list(metric_params)
        else:
            self.metric_funcs = None

    def preprocessing_input(self, images, targets=None):
        if self.batch_image is not None:
            images = self.batch_image(images, None)[0]
        return images, targets

    def calculate_sigmoid_losses(self, targets, seg_preds, image_shape):
        device = seg_preds[0].device
        ytrue = torch.stack([t['masks'].value for t in targets],dim=0).long()
        losses = {}
        if isinstance(seg_preds, list) | isinstance(seg_preds, tuple):
            for nb_level, (preds, loss_func_list, loss_w_list) in enumerate(zip(seg_preds)):
                stride = image_shape[-1]/preds.shape[-1]
                gt_boxes_mask = preds.new_full(preds.shape,0,device=preds.device)
                gt_boxes_cr = []
                gt_boxes_xxyy = []
                for n_img, target in enumerate(targets, self.loss_funcs, self.loss_weights):
                    boxes = torch.round(target['bboxes'].value/stride).type(torch.int32)
                    labels = target['labels'].value
                    for n in range(len(labels)):
                        box = boxes[n,:]
                        c = labels[n].type(torch.int32)
                        gt_boxes_mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                        height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                        r  = torch.sqrt(height**2+width**2)
                        cx = (box[2]+box[0]+1)//2
                        cy = (box[3]+box[1]+1)//2
                        gt_boxes_cr.append(torch.tensor([n_img, c, cx, cy, r]))
                        gt_boxes_xxyy.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
                if len(gt_boxes_cr)==0:
                    gt_boxes_cr = torch.empty((0,5), device=device)
                else:
                    gt_boxes_cr = torch.stack(gt_boxes_cr, dim=0)
                if len(gt_boxes_xxyy)==0:
                    gt_boxes_xxyy = torch.empty((0,6), device=device)
                else:
                    gt_boxes_xxyy = torch.stack(gt_boxes_xxyy, dim=0) 

                assert gt_boxes_cr.shape[0] == gt_boxes_xxyy.shape[0]

                kwargs_opt = {'ypred':preds, 'ytrue':ytrue, 'gt_boxes_mask':gt_boxes_mask,
                            'gt_boxes_xxyy':gt_boxes_xxyy, 'gt_boxes_cr':gt_boxes_cr}
                for loss_func, loss_w in zip(loss_func_list, loss_w_list):
                    loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w, nb_level)
                    losses.update(loss_v)
        else:
            stride = image_shape[-1]/seg_preds.shape[-1]
            gt_boxes_mask = seg_preds.new_full(seg_preds.shape,0,device=seg_preds.device)
            gt_boxes_index = []
            gt_boxes_cr = []
            gt_boxes_xxyy = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target['bboxes'].value/stride).type(torch.int32)
                labels = target['labels'].value
                for n in range(len(labels)):
                    box = boxes[n,:]
                    c = labels[n].type(torch.int32)
                    gt_boxes_mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                    height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                    r  = torch.sqrt(height**2+width**2)
                    cx = (box[2]+box[0]+1)//2
                    cy = (box[3]+box[1]+1)//2
                    gt_boxes_index.append(torch.tensor([n_img, c]))
                    gt_boxes_cr.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes_xxyy.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
            if len(gt_boxes_index)==0:
                gt_boxes_index = torch.empty((0,2), device=device)
            else:
                gt_boxes_index = torch.stack(gt_boxes_index, dim=0)
            if len(gt_boxes_cr)==0:
                gt_boxes_cr = torch.empty((0,5), device=device)
            else:
                gt_boxes_cr = torch.stack(gt_boxes_cr, dim=0)
            if len(gt_boxes_xxyy)==0:
                gt_boxes_xxyy = torch.empty((0,6), device=device)
            else:
                gt_boxes_xxyy = torch.stack(gt_boxes_xxyy, dim=0) 

            assert gt_boxes_cr.shape[0] == gt_boxes_xxyy.shape[0]

            kwargs_opt = {'ypred': seg_preds, 'ytrue': ytrue, 'gt_boxes_mask': gt_boxes_mask,
                        'gt_boxes_xxyy': gt_boxes_xxyy, 'gt_boxes_cr': gt_boxes_cr,
                        'gt_boxes_index': gt_boxes_index}
            for loss_func, loss_w in zip(self.loss_funcs,self.loss_weights):
                loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w)
                losses.update(loss_v)
        return losses

    def calculate_softmax_losses(self, targets, seg_preds, image_shape):
        device = seg_preds[0].device
        ytrue = torch.stack([t['masks'].value for t in targets],dim=0).long()
        losses = dict()
        if isinstance(seg_preds, list) | isinstance(seg_preds, tuple):
            for nb_level, (preds, loss_func_list, loss_w_list) in enumerate(zip(seg_preds)):
                stride = image_shape[-1]/preds.shape[-1]
                gt_boxes_mask = preds.new_full(preds.shape,0,device=preds.device)
                crop_boxes = []
                gt_boxes   = []
                for n_img, target in enumerate(targets):
                    boxes = torch.round(target['bboxes'].value/stride).type(torch.int32)
                    labels = target['labels'].value
                    for n in range(len(labels)):
                        box = boxes[n,:]
                        c   = labels[n].type(torch.int32)
                        gt_boxes_mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                        height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                        r  = torch.sqrt(height**2+width**2)
                        cx = (box[2]+box[0]+1)//2
                        cy = (box[3]+box[1]+1)//2
                        # print('//// box ////',box, cx, cy, r)
                        crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                        gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
                if len(crop_boxes)==0:
                    crop_boxes = torch.empty((0,5), device=device)
                else:
                    crop_boxes = torch.stack(crop_boxes, dim=0)
                if len(gt_boxes)==0:
                    gt_boxes = torch.empty((0,6), device=device)
                else:
                    gt_boxes = torch.stack(gt_boxes, dim=0) 

                # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
                assert crop_boxes.shape[0]==gt_boxes.shape[0]

                kwargs_opt = {'ypred':preds, 'ytrue':ytrue, 'gt_boxes_mask':gt_boxes_mask,
                            'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes}
                for loss_func, loss_w in zip(loss_func_list, loss_w_list):
                    loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w, nb_level)
                    losses.update(loss_v)
        else:            
            stride = image_shape[-1]/seg_preds.shape[-1]
            gt_boxes_mask = seg_preds.new_full(seg_preds.shape, 0, device=seg_preds.device)
            crop_boxes = []
            gt_boxes   = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target['bboxes'].value/stride).type(torch.int32)
                labels = target['labels'].value
                for n in range(len(labels)):
                    box = boxes[n,:]
                    c   = labels[n].type(torch.int32)
                    gt_boxes_mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                    height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                    r  = torch.sqrt(height**2+width**2)
                    cx = (box[2]+box[0]+1)//2
                    cy = (box[3]+box[1]+1)//2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
            if len(crop_boxes)==0:
                crop_boxes = torch.empty((0,5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes)==0:
                gt_boxes = torch.empty((0,6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0) 

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0]==gt_boxes.shape[0]

            kwargs_opt = {'ypred':seg_preds, 'ytrue':ytrue, 'gt_boxes_mask':gt_boxes_mask,
                        'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes}
            for loss_func, loss_w in zip(self.loss_funcs, self.loss_weights):
                loss_v = self.calculate_loss(kwargs_opt, loss_func, loss_w)
                losses.update(loss_v)

        return losses

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
        if self.final_activation == 'softmax':
            losses = self.calculate_softmax_losses(targets, ypred, self.batch_image_shape)
        else:
            losses = self.calculate_sigmoid_losses(targets, ypred, self.batch_image_shape)
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
                    if pred.shape[-1] == targets['masks'].value.shape[-1]:
                        true = targets['masks'].value
                    else:
                        true = F.resize(targets['masks'].value, pred.shape[-2:], InterpolationMode.NEAREST)
                    kwargs_opt = {'ypred': pred, 'ytrue': true}
                    for metric_func in self.metric_funcs:
                        metric_v = self.calculate_metric(kwargs_opt, metric_func, nb_level)
                        metrics.update(metric_v)
            else:
                kwargs_opt = {'ypred': ypred, 'ytrue': targets['masks'].value}
                for metric_func in self.metric_funcs:
                    metric_v = self.calculate_metric(kwargs_opt, metric_func)
                    metrics.update(metric_v)
        return metrics