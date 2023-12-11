import torch
from torch import nn
from ..ops.misc import SqueezeExcitation
from ...builder import NETS


__all__ = ("VGG_SE")


cfgs = {
    'A': [64, 'se', 'M', 128, 'se', 'M', 256, 256, 'se', 'M', 512, 512, 'se', 'M', 512, 512, 'se', 'M'],
    'B': [64, 64, 'se', 'M', 128, 128, 'se', 'M', 256, 256, 'se', 'M', 512, 512, 'se', 'M', 512, 512, 'se', 'M'],
    'C': [32, 32, 'se', 'M', 64, 64, 'se', 'M', 128, 128, 'se', 'M', 256, 256, 'se', 'M', 256, 256, 'se', 'M'],
}


@NETS.register_module()
class VGG_SE(nn.Module):
    def __init__(self, cfg, input_dim, num_classes=2, featuremap=(7, 7),
                 batch_norm=True, ratio=16, init_weights=True):
        super(VGG_SE, self).__init__()
        self.features = make_layers(cfg, input_dim, batch_norm=batch_norm, ratio=ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(featuremap)
        channels_list = [c for c in cfg if not isinstance(c, str)]
        self.classifier = nn.Sequential(
            nn.Linear(channels_list[-1] * featuremap[0] * featuremap[1], 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, input_dim, batch_norm=False, ratio=16):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'se':
            layers += [SqueezeExcitation(in_channels, in_channels//ratio)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)