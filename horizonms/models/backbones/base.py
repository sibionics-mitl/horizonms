from collections import OrderedDict
from torch import nn
from ..necks.feature_pyramid_network import FeaturePyramidNetwork
from ...builder import BACKBONES


__all__ = ("IntermediateLayerGetter", "Backbone", "BackboneWithFPN")


@BACKBONES.register_module()
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model.

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features.
        return_layers (List): the returned layers.
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        # out = OrderedDict()
        out = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                # out[out_name] = x
                out.append(x)
        if len(out) == 1:
            out = out[0]
        return out


@BACKBONES.register_module()
class Backbone(nn.Module):
    r"""It extracts a submodel that returns the feature maps specified in return_layers.
    
    Args:
        backbone (nn.Module):
        return_layers (List): the returned layers.
        in_channels_list (List[int]): number of channels for each feature map that is returned.
    Attributes:
        out_channels (int): the number of channels in the returned layers.
    """
    def __init__(self, backbone, return_layers, in_channels_list):
        super(Backbone, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.out_channels = in_channels_list
        self.return_layers = return_layers
        if len(in_channels_list) == 1:
            self.out_channels = in_channels_list[0]

    def forward(self, x):
        x = self.body(x)
        return x


@BACKBONES.register_module()
class BackboneWithFPN(nn.Module):
    r"""It extracts a submodel that returns the feature maps specified in return_layers
    and adds a FPN on top of the submodel.
    
    Args:
        backbone (nn.Module):
        return_layers (List): the returned layers.
        in_channels_list (List[int]): the number of channels for each feature map that is returned.
        out_channels (int): the number of channels in the returned layers.
        pyramid_levels (int): the number of the levels of feature pyramids.
    """    
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, pyramid_levels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            pyramid_levels=pyramid_levels,
        )
        self.out_channels = out_channels

        # nb_params = sum(p.numel() for p in self.body.parameters() if p.requires_grad)
        # print('# trainable parameters in backbone: {}'.format(nb_params))
        # nb_params = sum(p.numel() for p in self.fpn.parameters() if p.requires_grad)
        # print('# trainable parameters in fpn: {}'.format(nb_params))

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x