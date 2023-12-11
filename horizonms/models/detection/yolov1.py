import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchors_yolo import generate_stride, generate_feature_shape, generate_target_yolov1
from .boxes import paired_box_iou
from torchvision.ops import boxes as box_ops
from ... import NETS, LOSSES, METRICS
from ... import build_backbone, build_head, build_neck


__all__ = ("YOLOv1", "YOLOv1Losses", "YOLOv1Metrics")


@NETS.register_module()
class YOLOv1(nn.Module):
    r"""Yolov1 from 
    J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You only look once: Unified, real-time object detection",
    In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 779-788, 2016.

    Args:
        backbone (Dict): the configuration of the backbone.
        neck (Dict): the configuration of the neck.
        head (Dict): the configuration of the head.
        stride (int): the number of stride of the network in the input.
    """
    def __init__(self, backbone, neck=None, head=None, stride=None):
        super(YOLOv1, self).__init__()
        assert isinstance(backbone, dict) | isinstance(backbone, nn.Module), \
            f"backbone has to be a dictionary for backbone parameters or a nn.Module for backbone, but got {type(backbone)}."
        assert isinstance(neck, dict) | isinstance(neck, nn.Module) | (neck is None), \
            f"net has to be a dictionary for backbone parameters, a nn.Module for net, or None, but got {type(neck)}."
        assert isinstance(head, dict) | isinstance(head, nn.Module), \
            f"head has to be a dictionary for head parameters or a nn.Module for head, but got {type(head)}."
        if isinstance(backbone, dict):
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = backbone
        if isinstance(neck, dict):
            self.neck = build_neck(neck)
        else:
            self.neck = neck
        if isinstance(head, dict):
            self.head = build_head(head)
        else:
            self.head = head

        nb_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f'# trainable parameters in backbone: {nb_params}')
        if self.neck is not None:
            nb_params = sum(p.numel() for p in self.neck.parameters() if p.requires_grad)
            print(f'# trainable parameters in neck: {nb_params}')
        nb_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f'# trainable parameters in head: {nb_params}')

        if not hasattr(self.head, "num_classes"):
            raise ValueError(
                "head should contain an attribute num_classes "
                "specifying the number of classes")
        if not hasattr(self.head, "num_boxes"):
            raise ValueError(
                "head should contain an attribute num_boxes "
                "specifying the number of boxes")

        self.stride = stride
        self.num_classes = self.head.num_classes
        self.num_boxes = self.head.num_boxes
        if hasattr(self.head, 'feature_shape'):
            self.feature_shape = self.head.feature_shape
        else:
            self.feature_shape = None
        # self.initialize_weights()

    def forward(self, x):
        """
        x = (classification, confidence, txtytwth)
        classification, [B, num_classes, H, W]
        confidence, [B, num_boxes, H, W]
        bboxes, [B, 4*num_boxes, H, W]
        """
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_stride(self, feature_shape, input_shape):
        return generate_stride(feature_shape, input_shape)

    def get_feature_shape(self, stride, input_shape):
        return generate_feature_shape(stride, input_shape)

    def get_gts(self, feature_shape, input_shape, targets):
        return generate_target_yolov1(feature_shape, input_shape, targets, self.num_boxes, self.num_classes)

    def nms_postprocessing(self, pred_boxes, pred_scores, pred_cls_inds,
                           image_shapes_list, nms_params, flags=None):
        results = []
        device = pred_boxes.device
        for bs in range(pred_boxes.shape[0]):
            boxes, scores, cls_inds = pred_boxes[bs], pred_scores[bs], pred_cls_inds[bs]
            if flags is not None:
                flag_conf = flags[bs]
                boxes = boxes[flag_conf, :]
                scores = scores[flag_conf]
                cls_inds = cls_inds[flag_conf]
            image_shape = torch.tensor(image_shapes_list[bs], dtype=torch.float32, device=device)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_all_boxes, image_all_scores, image_all_labels = [], [], []
            for n_cls in range(self.num_classes):
                # remove low scoring boxes
                flag = (cls_inds==n_cls) & (scores > nms_params['nms_score_threshold'])
                if flag.sum() == 0:
                    continue
                box = boxes[flag]
                sc  = scores[flag]
                label = cls_inds[flag]

                # remove empty boxes
                keep = box_ops.remove_small_boxes(box, min_size=1e-2)
                box, sc = box[keep], sc[keep]

                # non-maximum suppression
                keep = box_ops.nms(box, sc, nms_params['nms_iou_threshold'])
                keep = keep[:nms_params['detections_per_class']]
                box, sc, label = box[keep], sc[keep], label[keep]

                image_all_boxes.append(box)
                image_all_scores.append(sc)
                image_all_labels.append(label)

            if len(image_all_boxes)>0:
                image_all_boxes = torch.cat(image_all_boxes, dim=0)
                image_all_scores = torch.cat(image_all_scores, dim=0)
                image_all_labels = torch.cat(image_all_labels, dim=0)
            else:
                image_all_boxes = torch.empty((0,4), dtype=torch.float32, device=device)
                image_all_scores = torch.empty((0,1), dtype=torch.float32, device=device)
                image_all_labels = torch.empty((0,1), dtype=torch.float32, device=device)
            results.append({
                        "boxes":  image_all_boxes,
                        "labels": image_all_labels,
                        "scores": image_all_scores,
                    })
        return results


@LOSSES.register_module()
class YOLOv1Losses():
    def __init__(self, lambda_coord=5, lambda_obj=1, lambda_noobj=0.5):
        super(YOLOv1Losses, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def __call__(self, gts, preds, input_shape, box_coder):
        pred_cls, pred_confidence, pred_txtytwth = preds[0], preds[1], preds[2]
        gt_cls, gt_objectness, gt_txtytwth = gts[0], gts[1], gts[2]
        bs = pred_cls.shape[0]
        
        obj_mask = gt_objectness.sum(dim=1, keepdim=True)
        index = torch.nonzero(obj_mask>0.5, as_tuple=False)
        nobj = index.shape[0]

        gt_cls_pos = gt_cls[index[:,0], :, index[:,2], index[:,3]]
        pd_cls_pos = pred_cls[index[:,0], :, index[:,2], index[:,3]]

        pred_confidence_pos = pred_confidence[index[:,0], :, index[:,2], index[:,3]]

        gt_txtytwth_pos = gt_txtytwth[index[:,0], :, index[:,2], index[:,3]]
        pred_txtytwth_pos = pred_txtytwth[index[:,0], :, index[:,2], index[:,3]]
        gt_txtytwth_pos = gt_txtytwth_pos.reshape(nobj, -1, 4).permute(0, 2, 1).contiguous()
        pred_txtytwth_pos = pred_txtytwth_pos.reshape(nobj, -1, 4).permute(0, 2, 1).contiguous()
        
        gt_boxes = box_coder.decode(gt_txtytwth, input_shape)
        pred_boxes = box_coder.decode(pred_txtytwth, input_shape)
        iou = paired_box_iou(gt_boxes, pred_boxes)
        
        # keep only one "responsible" predictor
        iou_pos = iou[index[:,0], :, index[:,2], index[:,3]]
        idx_pos = iou_pos.argmax(dim=1).unsqueeze(dim=1)
        pred_confidence_pos_sel = pred_confidence_pos.gather(1,idx_pos)
        iou_pos_sel = iou_pos.gather(1,idx_pos)
        idx = idx_pos.unsqueeze(dim=2).repeat(1, 4, 1)
        pred_txtytwth_pos = pred_txtytwth_pos.gather(2,idx)
        gt_txtytwth_pos = gt_txtytwth_pos.gather(2,idx)

        flag = iou_pos != iou_pos.max(dim=1, keepdim=True)[0]
        pred_confidence_pos_notsel = pred_confidence_pos[flag]

        index_neg = torch.nonzero(obj_mask<=0.5, as_tuple=False)
        pred_confidence_neg = pred_confidence[index_neg[:,0], :, index_neg[:,2], index_neg[:,3]]
        gt_response_neg = gt_objectness[index_neg[:,0], :, index_neg[:,2], index_neg[:,3]]

        loss_cls_pos = F.mse_loss(pd_cls_pos, gt_cls_pos, size_average=False)/bs
        loss_box_pos = (F.mse_loss(pred_txtytwth_pos[:,:2], gt_txtytwth_pos[:,:2], size_average=False)
         + F.mse_loss(torch.sqrt(pred_txtytwth_pos[:,2:]), torch.sqrt(gt_txtytwth_pos[:,2:]), size_average=False))/bs * self.lambda_coord
        loss_confidence_pos_sel = 2*F.mse_loss(pred_confidence_pos_sel, iou_pos_sel.detach(), size_average=False)/bs * self.lambda_obj
        loss_confidence_neg = F.mse_loss(pred_confidence_neg, gt_response_neg, size_average=False) / bs * self.lambda_noobj
        loss_confidence_pos_notsel = F.mse_loss(pred_confidence_pos_notsel, torch.full_like(pred_confidence_pos_notsel, 0), size_average=False)/bs
        
        losses = {"loss_cls_pos": loss_cls_pos, "loss_box_pos": loss_box_pos,
            "loss_confidence_pos_sel": loss_confidence_pos_sel, "loss_confidence_pos_notsel": loss_confidence_pos_notsel,
            "loss_confidence_neg": loss_confidence_neg}
        
        return losses


@METRICS.register_module()
class YOLOv1Metrics():
    def __init__(self):
        super(YOLOv1Metrics, self).__init__()

    def __call__(self, gts, preds, input_shape, box_coder):
        pred_cls, pred_confidence, pred_txtytwth = preds[0], preds[1], preds[2]
        gt_cls, gt_objectness, gt_txtytwth = gts[0], gts[1], gts[2]
        bs = pred_cls.shape[0]
        
        obj_mask = gt_objectness.sum(dim=1, keepdim=True)
        index = torch.nonzero(obj_mask>0.5, as_tuple=False)
        nobj = index.shape[0]

        gt_cls_pos = gt_cls[index[:,0], :, index[:,2], index[:,3]]
        pd_cls_pos = pred_cls[index[:,0], :, index[:,2], index[:,3]]

        pred_confidence_pos = pred_confidence[index[:,0], :, index[:,2], index[:,3]]

        gt_txtytwth_pos = gt_txtytwth[index[:,0], :, index[:,2], index[:,3]]
        pred_txtytwth_pos = pred_txtytwth[index[:,0], :, index[:,2], index[:,3]]
        gt_txtytwth_pos = gt_txtytwth_pos.reshape(nobj, -1, 4).permute(0, 2, 1).contiguous()
        pred_txtytwth_pos = pred_txtytwth_pos.reshape(nobj, -1, 4).permute(0, 2, 1).contiguous()
        
        gt_boxes = box_coder.decode(gt_txtytwth, input_shape)
        pred_boxes = box_coder.decode(pred_txtytwth, input_shape)
        iou = paired_box_iou(gt_boxes, pred_boxes)
        
        # keep only one "responsible" predictor
        iou_pos = iou[index[:,0], :, index[:,2], index[:,3]]
        idx_pos = iou_pos.argmax(dim=1).unsqueeze(dim=1)
        pred_confidence_pos = pred_confidence_pos.gather(1,idx_pos)
        iou_pos = iou_pos.gather(1,idx_pos)
        idx = idx_pos.unsqueeze(dim=2).repeat(1, 4, 1)
        pred_txtytwth_pos = pred_txtytwth_pos.gather(2,idx)
        gt_txtytwth_pos = gt_txtytwth_pos.gather(2,idx)

        index_neg = torch.nonzero(obj_mask<=0.5, as_tuple=False)
        pred_confidence_neg = pred_confidence[index_neg[:,0], :, index_neg[:,2], index_neg[:,3]]
        gt_response_neg = gt_objectness[index_neg[:,0], :, index_neg[:,2], index_neg[:,3]]

        loss_cls_pos = F.mse_loss(gt_cls_pos, pd_cls_pos, reduction='mean')
        loss_box_pos = F.mse_loss(gt_txtytwth_pos[:,:2], pred_txtytwth_pos[:,:2], reduction='mean') + \
            F.mse_loss(torch.sqrt(gt_txtytwth_pos[:,2:]), torch.sqrt(pred_txtytwth_pos[:,2:]), reduction='mean')
        loss_confidence_pos = F.mse_loss(pred_confidence_pos, iou_pos, reduction='mean')
        loss_confidence_neg = F.mse_loss(pred_confidence_neg, gt_response_neg, reduction='mean')

        pd_cls_label = pd_cls_pos[gt_cls_pos > 0]
        metrics = {"loss_cls_pos_avg": loss_cls_pos, "loss_box_pos_avg": loss_box_pos,
            "loss_confidence_pos_avg": loss_confidence_pos, "loss_confidence_neg_avg": loss_confidence_neg,
            "iou_pos_avg": iou_pos.mean(), "pd_cls_label": pd_cls_label.mean()}
        return metrics
