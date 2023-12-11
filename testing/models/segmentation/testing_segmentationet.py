import os, sys
sys.path.insert(0, os.getcwd())
import torch
from horizonms.models.segmentation import get_segmentation_net


if __name__ == "__main__":
    cfg = dict(name="ResidualUNet", input_dim=3, num_classes=5)
    net = get_segmentation_net(cfg)
    print(net)

    x = torch.randn(2,3,224,224)
    y = net(x)
    print(x.shape, y.shape)