from collections import OrderedDict
import torch
from torch import nn
import math
from .anchors_retinanet import AnchorGenerator, anchor_targets_bbox
from torchvision.ops import boxes as box_ops
from ...builder import NETS, HEADS
from ... import build_backbone, build_neck, build_head


__all__ = ("Retinanet", "RetinanetHead")


@NETS.register_module()
class Retinanet(nn.Module):
    r"""Retinanet from 
    T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection", 
    In Proceedings of the IEEE international conference on computer vision, pp. 2980-2988, 2017.

    Args:
        backbone (Dict): the configuration of the backbone.
        neck (Dict): the configuration of the neck.
        head (Dict): the configuration of the head.
        num_classes (int): the number of classes.
        anchor_params (Dict): the parameters of anchors.
        nms_params (Dict): the parameters of NMS.
    """
    def __init__(self, backbone, neck=None, head=None, num_classes=None,
                 # anchor parameters
                 anchor_params= dict(anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
                        anchor_aspect_ratios=((0.5, 1.0, 2.0),) * 5, 
                        anchor_scales=((2**0, 2**(1/3), 2**(2/3)),) * 5),
                 nms_params=dict(nms_score_threshold=0.005, nms_iou_threshold=0.05,
                                 detections_per_class=10)
                 ):
        super(Retinanet, self).__init__()

        self.nms_params = nms_params
        if isinstance(backbone, dict):
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = backbone
        out_channels = self.backbone.out_channels

        if not hasattr(self.backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        if isinstance(neck, dict):
            in_channels_list = neck.get("in_channels_list", None)
            if in_channels_list is not None:
                if in_channels_list != out_channels:
                    print(f"in_channels_list was wrongly given as {in_channels_list} in neck, "
                          f"it was reset as {out_channels}.")
            neck.update({"in_channels_list": out_channels})
            self.neck = build_neck(neck)
        else:
            self.neck = neck

        if self.neck is not None:
            out_channels = self.neck.out_channels

        self.anchor_generator = AnchorGenerator(**anchor_params)
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.num_anchors = self.anchor_generator.num_anchors_per_location()[0]

        if isinstance(head, dict):
            in_channels = head.get("in_channels", None)
            if in_channels is not None:
                if in_channels != out_channels:
                    print(f"in_channels was wrongly given as {in_channels} in head, "
                          f"it was reset as {out_channels}.")
            head.update(dict(in_channels=out_channels, num_anchors=num_anchors,
                num_classes=num_classes, num_fpn=len(anchor_params['anchor_sizes'])))
            self.head = build_head(head)
        elif head is None:
            self.head = RetinanetHead(
                out_channels, num_anchors, num_conv=4,
                feature_size=256, num_classes=num_classes,
                num_fpn=len(anchor_params['anchor_sizes']), prior=0.01)
        else:
            self.head = head

        if not hasattr(self.head, "num_classes"):
            raise ValueError(
                "head should contain an attribute num_classes "
                "specifying the number of classes")
        self.num_classes = self.head.num_classes

        nb_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f'# trainable parameters in backbone: {nb_params}')
        if self.neck is not None:
            nb_params = sum(p.numel() for p in self.neck.parameters() if p.requires_grad)
            print(f'# trainable parameters in neck: {nb_params}')
        nb_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f'# trainable parameters in head: {nb_params}')    

    def forward(self, images, image_sizes_list=None):
        features = self.backbone(images)
        if self.neck is not None:
            features = self.neck(features)
        pred_cls, pred_bbox_deltas = self.head(features)
        if image_sizes_list is None:
            return [pred_cls, pred_bbox_deltas]
        else:
            anchors = self.anchor_generator(features, images.shape[-2:], image_sizes_list)
            return [pred_cls, pred_bbox_deltas], anchors

    def get_gts(self, targets, anchors, original_image_sizes, box_coder, bbox_thresholds):
        ## calculate losses
        regression_targets, labels_targets, anchor_state = \
                                    anchor_targets_bbox(anchors, targets, self.num_classes, original_image_sizes, 
                                    negative_overlap=bbox_thresholds['bbox_low_iou_thresh'],
                                    positive_overlap=bbox_thresholds['bbox_high_iou_thresh'])
        regression_targets = box_coder.encode(torch.stack(anchors, dim=0), regression_targets)
        return [labels_targets, regression_targets, anchor_state]

    def pred_postprocessing_for_loss_calculation(self, box_preds):
        box_cls, box_regression = box_preds
        box_cls_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
        ):
            N, AxC, H, W = box_cls_per_level.shape
            Ax4 = box_regression_per_level.shape[1]
            A = Ax4 // 4
            C = AxC // A
            box_cls_per_level = permute_and_flatten(
                box_cls_per_level, N, A, C, H, W
            )
            box_cls_flattened.append(box_cls_per_level)

            box_regression_per_level = permute_and_flatten(
                box_regression_per_level, N, A, 4, H, W
            )
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        box_cls = torch.cat(box_cls_flattened, dim=1)#.flatten(0, -2)
        box_regression = torch.cat(box_regression_flattened, dim=1)#.reshape(-1, 4)
        return box_cls, box_regression

    def postprocess_detections(self, preds, anchors, image_shapes, box_coder):
        device = preds[0].device
        num_classes = preds[0].shape[-1]
        pred_boxes = box_coder.decode(torch.stack(anchors, dim=0), preds[1])

        result = []
        for bs in range(pred_boxes.shape[0]):
            boxes, scores, image_shape = pred_boxes[bs], preds[0][bs], image_shapes[bs]
            image_shape = torch.tensor(image_shape, dtype=torch.float32, device=device)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_all_boxes = []
            image_all_scores = []
            image_all_labels = []

            for n_cls in range(num_classes):
                # remove low scoring boxes
                inds = torch.where(scores[:,n_cls] > self.nms_params['nms_score_threshold'])[0]
                if len(inds)==0:
                    # print(scores[:,n_cls].max())
                    # print('no detections for label = {}'.format(n_cls))
                    continue
                box = boxes[inds, :]
                sc  = scores[inds, n_cls]
                # print('candidata box {:d}'.format(len(inds)))

                # remove empty boxes
                keep = box_ops.remove_small_boxes(box, min_size=1e-2)
                box, sc = box[keep], sc[keep]

                # non-maximum suppression
                keep = box_ops.nms(box, sc, self.nms_params['nms_iou_threshold'])
                keep = keep[:self.nms_params['detections_per_class']]
                box, sc = box[keep], sc[keep]
                labels = torch.tensor([n_cls]*len(sc), dtype=torch.int, device=device)

                image_all_boxes.append(box)
                image_all_scores.append(sc)
                image_all_labels.append(labels)
            if len(image_all_boxes)>0:
                image_all_boxes = torch.cat(image_all_boxes, dim=0)
                image_all_scores = torch.cat(image_all_scores, dim=0)
                image_all_labels = torch.cat(image_all_labels, dim=0)
            else:
                image_all_boxes = torch.empty((0,4), dtype=torch.float32, device=device)
                image_all_scores = torch.empty((0,1), dtype=torch.float32, device=device)
                image_all_labels = torch.empty((0,1), dtype=torch.float32, device=device)
            result.append({
                        "boxes":  image_all_boxes,
                        "labels": image_all_labels,
                        "scores": image_all_scores,
                    })

        return result


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2).contiguous()
    layer = layer.reshape(N, -1, C)
    return layer


@HEADS.register_module()
class RetinanetHeadSingle(nn.Module):
    def __init__(self, in_channels, num_anchors, num_conv=4, feature_size=256, num_classes=81, prior=0.01):
        super(RetinanetHeadSingle, self).__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.num_conv = num_conv
        self.cls_feat = self._make_layers()
        self.bbox_feat = self._make_layers()
        self.cls_pred = nn.Sequential(OrderedDict([
                                      ('cls',    nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)),
                                      ('sigmod', nn.Sigmoid())
                                    ]))
        self.bbox_pred = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        bias_value = -(math.log((1 - prior) / prior))
        self.cls_pred.cls.bias.data.fill_(bias_value)

    def _make_layers(self):
        in_channels  = self.in_channels
        feature_size = self.feature_size
        num_conv     = self.num_conv
        layers = OrderedDict()
        layers['conv1'] = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        layers['relu1'] = nn.ReLU()
        for k in range(1,num_conv):
            layers['conv'+str(k+1)] = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
            layers['relu'+str(k+1)] = nn.ReLU()
        return nn.Sequential(layers)

    def forward(self, x):
        cls_feat = self.cls_feat(x)
        bbox_feat = self.bbox_feat(x)
        cls_pred = self.cls_pred(cls_feat)
        bbox_reg = self.bbox_pred(bbox_feat)
        return cls_pred, bbox_reg


@HEADS.register_module()
class RetinanetHead(nn.Module):
    r"""Head of Retinanet from 
    T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection", 
    In Proceedings of the IEEE international conference on computer vision, pp. 2980-2988, 2017.
    """
    def __init__(self, in_channels, num_anchors, num_conv=4, feature_size=256, num_classes=81, num_fpn=4, prior=0.01):
        super(RetinanetHead, self).__init__()
        self.num_classes = num_classes
        self.heads_list = nn.ModuleList()
        for k in range(num_fpn):
            single_head = RetinanetHeadSingle(in_channels, num_anchors, num_conv, feature_size, num_classes, prior)
            self.heads_list.append(single_head)
       
    def forward(self, inputs):
        assert len(self.heads_list)==len(inputs)
        cls_preds, bbox_reg = [], []
        for x, heads in zip(inputs, self.heads_list):
            v_cls, v_reg = heads(x)
            cls_preds.append(v_cls)
            bbox_reg.append(v_reg)
        return cls_preds, bbox_reg