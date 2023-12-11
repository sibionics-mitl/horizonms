import os, sys
sys.path.insert(0, os.getcwd())
import torch
from horizonms.builder import build_backbone


if __name__ == "__main__":
    cfg = dict(name="resnet_unet_backbone", backbone_name="resnet18")
    cfg = dict(name="vgg_unet_backbone", backbone_name="vgg11_bn", trainable_stages=3)
    cfg = dict(name="densenet_unet_backbone", backbone_name="densenet121")
    cfg = dict(name="mobilenetv2_unet_backbone")
    net = build_backbone(cfg)
    print(net)
    print(net.return_layers)

    x = torch.randn(2,3,224,224)
    y = net(x)
    print(x.shape)
    print(type(y), [v.shape for v in y])

    # from horizonms.models.backbones import densenet_unet_backbone
    # net = densenet_unet_backbone("densenet121")
