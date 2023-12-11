import torch.nn as nn
from .base import IntermediateLayerGetter
from ...builder import BACKBONES


__all__ = ("VGGCfg", "BackboneVGGCfg", "vgg_cfg_backbone_v1", "vgg_cfg_backbone_v2")


CFG_A = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 
         512, 512, 'M', 512, 512, 'M', 1024]


class Conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2d_bn, self).__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2d_bn = nn.Sequential()
        self.conv2d_bn.add_module('conv2d', conv2d)
        self.conv2d_bn.add_module('bn', nn.BatchNorm2d(out_channels))
        self.conv2d_bn.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv2d_bn(x)
        return x


class VGGCfg(nn.Module):
    def __init__(self, input_dim, cfg=CFG_A):
        super(VGGCfg, self).__init__()
        # self.features = self.make_layers(cfg, input_dim)
        self.backbone = nn.Sequential()
        in_channels = input_dim
        for i, v in enumerate(cfg):
            if v == 'M':
                layer = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                layer = Conv2d_bn(in_channels, v)
                in_channels = v
            self.backbone.add_module(f'layer{i}', layer)
        if cfg[-1] == 'M':
            self.backbone.out_channels = cfg[-2]
        else:
            self.backbone.out_channels = cfg[-1]
        self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
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


@BACKBONES.register_module()
class BackboneVGGCfg(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list):
        super(BackboneVGGCfg, self).__init__()
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.out_channels = in_channels_list[-4:]

    def forward(self, images):
        x = self.backbone(images)
        return x


@BACKBONES.register_module()
def vgg_cfg_backbone_v1(input_dim=3, cfg=CFG_A, return_stages=5):
    backbone = VGGCfg(input_dim, cfg)
    return_index = [k-1  for k, param in enumerate(cfg) if param=='M'] + [len(cfg)-1]
    assert len(return_index) >= return_stages, f"return_stages has to be " \
        f"less than the available stages, but got return_stages = {return_stages}, " \
        f"and available stages = {len(return_index)}"
    return_index = return_index[-return_stages:]
    return_layers = {f"layer{index}": str(k) for k, index in enumerate(return_index)}
    in_channels_list = [cfg[index] for index in return_index]   
    return backbone.backbone, return_layers, in_channels_list


@BACKBONES.register_module()
def vgg_cfg_backbone_v2(input_dim=3, cfg=CFG_A, return_stages=5):
    backbone = VGGCfg(input_dim, cfg)
    return_index = [k  for k, param in enumerate(cfg) if param=='M']
    assert len(return_index) >= return_stages, f"return_stages has to be " \
        f"less than the available stages, but got return_stages = {return_stages}, " \
        f"and available stages = {len(return_index)}"
    return_index = return_index[-return_stages:]
    return_layers = {f"layer{index}": str(k) for k, index in enumerate(return_index)}
    in_channels_list = [cfg[index-1] for index in return_index]
    return BackboneVGGCfg(backbone.backbone, return_layers, in_channels_list)