import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from horizonms import build_net, build_detector, build_backbone, build_neck, \
           build_head, build_loss, build_metric
from horizonms import MODELS, HEADS, BACKBONES, NECKS


if __name__ == "__main__":
    # print all registered modules
    print(HEADS.__repr__)
    print(NECKS.__repr__)

    cfg_backbone = dict(
        name="efficientnet_backbone",
        backbone_name="efficientnet_b1",
        return_stages=1,
        pretrained=True,
        trainable_stages=7,
        )
    backbone = build_backbone(cfg_backbone)
    in_channels = backbone.out_channels
    # print(backbone)

    cfg_neck = dict(
        name="ClassificationPoolingNecks",
    )
    neck = build_neck(cfg_neck)
    if hasattr(neck, "out_channels"):
        in_channels = neck.out_channels
        print("---in_channels in neck---", in_channels)
    # print(neck)

    cfg_head = dict(
        name="ClassificationMultiHeads",
        input_dim=in_channels,
        dropout=0.5,
        num_softmax_classes_list=[1,2,3,4],
        num_sigmoid_classes_list=[53,44]
        )
    head = build_head(cfg_head)
    # print(head)

    x = torch.randn((2,3,224,224))
    b = backbone(x)
    n = neck(b)
    h = head(n)
    print(x.shape, b.shape, n.shape)
    print([v.shape for v in h])
    
    
    