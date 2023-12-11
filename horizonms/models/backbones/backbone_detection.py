from torch import nn
from ..necks.feature_pyramid_network import FeaturePyramidNetwork
from .base import IntermediateLayerGetter
from .backbone_vgg_cfg import vgg_cfg_backbone_v1, vgg_cfg_backbone_v2
from ...builder import BACKBONES, build_backbone


__all__ = ("vgg_fpn_det_v1", "vgg_fpn_det_v2")


@BACKBONES.register_module()
class DefaultRetinaNetBackbone(nn.Module):
    r"""It extracts the default backbone with FPN for RetinaNet.
    
    Args:
        backbone (nn.Module):
        return_layers (List): the returned layers.
        in_channels_list (List[int]): the list of the number of channels for inputs.
        fpn_out_channels (int): the number of channels of FPN output.
        pyramid_levels (List[int]): the levels of FPN.
    """  

    def __init__(self, backbone, return_layers, in_channels_list, 
                 fpn_out_channels=256, pyramid_levels=[2,3,4,5]):
        super(DefaultRetinaNetBackbone, self).__init__() 
        self.levels = len(pyramid_levels)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        nb_params = sum(p.numel() for p in self.body.parameters() if p.requires_grad)
        print('# trainable parameters in backbone: {}'.format(nb_params))

        fpn_in_channels_list = in_channels_list[-4:]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            pyramid_levels=pyramid_levels,
        )
        self.out_channels = fpn_out_channels
        nb_params = sum(p.numel() for p in self.fpn.parameters() if p.requires_grad)
        print('# trainable parameters in fpn: {}'.format(nb_params))

    def forward(self, images):
        x = self.body(images)
        x = self.fpn(x[-self.levels:])
        return x


@BACKBONES.register_module()
def vgg_fpn_det_v1(input_dim, backbone_cfg, backbone_version='BackboneWithFPN', 
        fpn_out_channels=256, pyramid_levels=[2,3,4,5]):
    backbone, return_layers, in_channels_list = vgg_cfg_backbone_v1(input_dim, 
                backbone_cfg, return_stages=len(pyramid_levels))
    cfg = dict(name=backbone_version, backbone=backbone, 
               return_layers=return_layers, in_channels_list=in_channels_list, 
               out_channels=fpn_out_channels, pyramid_levels=pyramid_levels)
    model = build_backbone(cfg)
    return model


@BACKBONES.register_module()
def vgg_fpn_det_v2(input_dim, backbone_cfg, backbone_version='BackboneWithFPN',
        fpn_out_channels=256, pyramid_levels=[2,3,4,5]):
    backbone, return_layers, in_channels_list = vgg_cfg_backbone_v2(input_dim, 
                backbone_cfg, return_stages=len(pyramid_levels))
    cfg = dict(name=backbone_version, backbone=backbone, 
               return_layers=return_layers, in_channels_list=in_channels_list, 
               out_channels=fpn_out_channels, pyramid_levels=pyramid_levels)
    model = build_backbone(cfg)
    return model