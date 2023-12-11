from torch import nn
from horizonms.models.nets import efficientnet
from ..nets import resnet, vgg, efficientnet
from .base import Backbone
from ...builder import BACKBONES


__all__ = ("vgg_backbone", "resnet_backbone", "efficientnet_backbone")


@BACKBONES.register_module()
def vgg_backbone(backbone_name, return_stages=5, input_dim=3, pretrained=False, 
                 model_dir='.', trainable_stages=None, **kwargs):
    r"""It extracts a backbone from VGG network.
    
    Args:
        backbone_name (str): the name of backbone.
        return_stages (int): the number of stages to be extracted.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_stages (int): the number of trainable (not frozen) stages starting from final stage.
    """   
    vgg_names = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                 'vgg16_bn', 'vgg19_bn', 'vgg19']
    if backbone_name not in vgg_names:
        raise ValueError(f"backbone name is wrong, it has to be in {vgg_names}")
    backbone = vgg.__dict__[backbone_name](input_dim, pretrained=pretrained, 
                                           model_dir=model_dir, **kwargs)
    backbone = backbone.features
    all_stages, stage = [], []
    for name, layer in backbone.named_modules():
        stage.append(name)
        if isinstance(layer, nn.MaxPool2d):
            all_stages.append(stage)
            stage = []
    all_stages = all_stages[::-1]
    if trainable_stages is None:
        trainable_stages = 5
    assert return_stages <= 5 and return_stages >= 1
    assert trainable_stages <= 5 and trainable_stages >= 0
    # assert (trainable_stages < 5) & pretrained, "When trainable_stages < 5, pretrained has to be True"
    stages_to_train = all_stages[:trainable_stages]

    # freeze layers
    layers_to_train = [v for stage in stages_to_train for v in stage]
    if pretrained and (trainable_stages < 5):
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

    all_return_stages = {cand[-1]: f"stage{k+1}" for k, cand in enumerate(all_stages)}
    all_stage_index = [stage[-1] for stage in all_stages]
    if 'narrow' in backbone_name:
        all_in_channels_list = [32, 64, 128, 256, 256]
    else:
        all_in_channels_list = [64, 128, 256, 512, 512]
    return_layers = {k: v for k, v in all_return_stages.items() 
                        if k in all_stage_index[:return_stages]}
    return Backbone(backbone, return_layers, all_in_channels_list[-return_stages:])


@BACKBONES.register_module()
def resnet_backbone(backbone_name, return_stages=4, input_dim=3, pretrained=False,
                    model_dir='.', trainable_stages=None, **kwargs):
    r"""It extracts a backbone from ResNet network.
    
    Args:
        backbone_name (str): the name of backbone.
        return_stages (int): the number of stages to be extracted.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_stages (int): the number of trainable (not frozen) stages starting from final stage.
    """   
    resnet_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 
                    'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
    if backbone_name not in resnet_names:
        raise ValueError(f"backbone name is wrong, it has to be in {resnet_names}")
    backbone = resnet.__dict__[backbone_name](input_dim, pretrained=pretrained,
                                              model_dir=model_dir, **kwargs)
    all_stages = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']
    if trainable_stages is None:
        trainable_stages = 5
    assert return_stages <= 5 and return_stages >= 1
    assert trainable_stages <= 5 and trainable_stages >= 0
    if (trainable_stages < 5):
        assert pretrained, "When trainable_stages < 5, pretrained has to be True"
    stages_to_train = all_stages[:trainable_stages]

    # freeze layers
    if pretrained and (trainable_stages < 5):
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in stages_to_train]):
                parameter.requires_grad_(False)

    all_return_stages = {'conv1': 'stage1', 'layer1': 'stage2',
                'layer2': 'stage3', 'layer3': 'stage4', 'layer4': 'stage5'}
    in_channels_stage2 = backbone.inplanes // 8
    all_in_channels_list = [64, in_channels_stage2, in_channels_stage2*2,
                            in_channels_stage2*4, in_channels_stage2*8]
    return_layers = {k: v for k,v in all_return_stages.items()
                        if k in all_stages[:return_stages]}
    return Backbone(backbone, return_layers, all_in_channels_list[-return_stages:])


@BACKBONES.register_module()
def efficientnet_backbone(backbone_name, return_stages=9, input_dim=3, pretrained=False, 
                 model_dir='.', trainable_stages=None, **kwargs):
    r"""It extracts a backbone from EfficientNet network.
    
    Args:
        backbone_name (str): the name of backbone.
        return_stages (int): the number of stages to be extracted.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_stages (int): the number of trainable (not frozen) stages starting from final stage.
    """  
    efficientnet_names = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6",
        "efficientnet_b7", "efficientnet_b8", "efficientnet_l2"]
    if backbone_name not in efficientnet_names:
        raise ValueError(f"backbone name is wrong, it has to be in {efficientnet_names}")
    backbone = efficientnet.__dict__[backbone_name](input_dim, pretrained=pretrained, 
                                           model_dir=model_dir, **kwargs)
    backbone = backbone.features
    all_stages = []
    for name, layer in backbone.named_modules():
        if len(name) == 1:
            all_stages.append(name)
    all_stages = all_stages[::-1]
    if trainable_stages is None:
        trainable_stages = 9
    assert return_stages <= 9 and return_stages >= 1
    assert trainable_stages <= 9 and trainable_stages >= 0
    stages_to_train = all_stages[:trainable_stages]

    params_dict = {
        'efficientnet_b0': 1.0,
        'efficientnet_b1': 1.0,
        'efficientnet_b2': 1.1,
        'efficientnet_b3': 1.2,
        'efficientnet_b4': 1.4,
        'efficientnet_b5': 1.6,
        'efficientnet_b6': 1.8,
        'efficientnet_b7': 2.0,
        'efficientnet_b8': 2.2,
        'efficientnet_l2': 4.3,
    }

    # freeze layers
    layers_to_train = [v for stage in stages_to_train for v in stage]
    if pretrained and (trainable_stages < 9):
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

    all_return_stages = {cand[-1]: f"stage{k+1}" for k, cand in enumerate(all_stages)}
    all_stage_index = [stage[-1] for stage in all_stages]
    all_in_channels_list = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
    all_in_channels_list = [int(s*params_dict[backbone_name]) for s in all_in_channels_list]
    return_layers = {k: v for k, v in all_return_stages.items() 
                        if k in all_stage_index[:return_stages]}
    return Backbone(backbone, return_layers, all_in_channels_list[-return_stages:])