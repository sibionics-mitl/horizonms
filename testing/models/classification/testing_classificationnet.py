import os, sys
sys.path.insert(0, os.getcwd())
import torch
from horizonms.models.classification import get_classification_net


if __name__ == "__main__":
    x = torch.randn(2,3,224,224)
    n_cfg = 2
    if n_cfg == 1:
        cfg = dict(backbone=dict(
                            name="efficientnet_backbone",
                            backbone_name="efficientnet_b1",
                            return_stages=1,
                            pretrained=True,
                            trainable_stages=7,
                   ), 
                   neck=dict(name="ClassificationPoolingNecks"), 
                   head=dict(
                            name="ClassificationMultiHeads",
                            dropout=0.5,
                            num_softmax_classes_list=[1,2,3,4],
                            num_sigmoid_classes_list=[53,44]
                   )
        )
    elif n_cfg == 2:
        cfg = dict(name="efficientnet_mixmultiheads_b1",
                num_softmax_classes_list=[1,2,3,4],
                num_sigmoid_classes_list=[53,44]
        )

    net = get_classification_net(cfg)
    y = net(x)
    print(net)
    print(x.shape, [v.shape for v in y])

    