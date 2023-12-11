from torch import nn
from ..nets import vgg, resnet#, densenet, mobilenet
from .base import BackboneWithFPN
from ...builder import BACKBONES


__all__ = ("vgg_fpn_backbone", "resnet_fpn_backbone")


@BACKBONES.register_module()
def vgg_fpn_backbone(backbone_name, input_dim=3, pretrained=False, model_dir='.',
                     pyramid_levels=[2, 3, 4, 5, 6, 7], 
                     trainable_stages=5, fpn_out_channels=256, **kwargs):
    r"""It extracts a backbone from VGG network and adds FPN to the backbone.
    
    Args:
        backbone_name (str): the name of backbone.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        pyramid_levels (List[int]): the levels of FPN.
        trainable_stages (int): number of trainable (not frozen) stages starting from final stage.
            Valid values are between 1 and 5, with 5 meaning all backbone layers are trainable.
        fpn_out_channels (int): the number of channels for the FPN output.
    """  
    vgg_names = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
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
    assert trainable_stages <= 5 and trainable_stages >= 1
    if (trainable_stages < 5):
        assert pretrained, "When trainable_stages < 5, pretrained has to be True"
    stages_to_train = all_stages[:trainable_stages]

    # freeze layers
    layers_to_train = [v for stage in stages_to_train for v in stage]
    if pretrained and (trainable_stages < 5):
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

    return_stages_index = all_stages[:4][::-1]
    return_stages = {cand[-1]: f"stage{k+1}" for k, cand in enumerate(return_stages_index)}
    if 'narrow' in backbone_name:
        in_channels_list = [64, 128, 256, 256]
    else:
        in_channels_list = [128, 256, 512, 512]
    return BackboneWithFPN(backbone, return_stages, in_channels_list, fpn_out_channels, pyramid_levels)


@BACKBONES.register_module()
def resnet_fpn_backbone(backbone_name, input_dim=3, pretrained=False, model_dir='.',
                        pyramid_levels=[2, 3, 4, 5, 6, 7], 
                        trainable_stages=None, fpn_out_channels=256, **kwargs):
    r"""It extracts a backbone from VGG network and adds FPN to the backbone.
    
    Args:
        backbone_name (str): the name of backbone.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        pyramid_levels (List[int]): the levels of FPN.
        trainable_stages (int): number of trainable (not frozen) stages starting from final stage.
            Valid values are between 1 and 5, with 5 meaning all backbone layers are trainable.
        fpn_out_channels (int): the number of channels for the FPN output.
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
    assert trainable_stages <= 5 and trainable_stages >= 1
    if (trainable_stages < 5):
        assert pretrained, "When trainable_stages < 5, pretrained has to be True"
    stages_to_train = all_stages[:trainable_stages]

    # freeze layers
    if pretrained and (trainable_stages < 5):
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in stages_to_train]):
                parameter.requires_grad_(False)

    return_layers = {'layer1': 'stage2', 'layer2': 'stage3',
                     'layer3': 'stage4', 'layer4': 'stage5'}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2, in_channels_stage2*2,
                        in_channels_stage2*4, in_channels_stage2*8]
    return BackboneWithFPN(backbone, return_layers, in_channels_list, fpn_out_channels, pyramid_levels)



# def densenet_fpn_backbone(backbone_name, input_dim=3, pyramid_levels=[2,3,4,5,6,7], pretrained=False, model_dir='.',
#                      trainable_layers=5, fpn_out_channels=256):
#     densenet_names = ['densenet121', 'densenet169', 'densenet201', 'densenet161']
#     if backbone_name not in densenet_names:
#         raise ValueError(f"backbone name is wrong, it has to be in {densenet_names}")

#     backbone = densenet.__dict__[backbone_name](input_dim, pretrained=pretrained, model_dir=model_dir)
#     backbone = backbone.features

#     # select layers that wont be frozen
#     assert trainable_layers <= 5 and trainable_layers >= 0
#     layers_to_train = ['denseblock4', 'denseblock3', 'denseblock2', 'denseblock1', 'conv0'][:trainable_layers]
#     # freeze layers only if pretrained backbone is used
#     for name, parameter in backbone.named_parameters():
#         if all([not name.startswith(layer) for layer in layers_to_train]):
#             parameter.requires_grad_(False)

#     return_layers = {'denseblock1': '0', 'denseblock2': '1', 'denseblock3': '2', 'denseblock4': '3'}
#     if backbone_name=='densenet121':
#         in_channels_list = [256,512,1024,1024]
#     elif backbone_name=='densenet169':
#         in_channels_list = [256,512,1280,1664]
#     elif backbone_name=='densenet201':
#         in_channels_list = [256,512,1792,1920]
#     elif backbone_name=='densenet161':
#         in_channels_list = [384,768,2112,2208]
#     return BackboneWithFPN(backbone, return_layers, in_channels_list, fpn_out_channels, pyramid_levels)

# def mobilenetv2_fpn_backbone(backbone_name, input_dim=3, pyramid_levels=[2,3,4,5,6,7], pretrained=False, model_dir='.',
#                      trainable_layers=19, fpn_out_channels=256):
    
#     backbone = mobilenet.__dict__[backbone_name](input_dim, pretrained=pretrained, model_dir=model_dir)
#     backbone = backbone.features
#     # print(backbone)

#     # # select layers that wont be frozen
#     assert trainable_layers <= 19 and trainable_layers >= 0
#     layers_to_train = [str(k) for k in range(19)][:trainable_layers]
#     # freeze layers only if pretrained backbone is used
#     for name, parameter in backbone.named_parameters():
#         if all([not name.startswith(layer) for layer in layers_to_train]):
#             parameter.requires_grad_(False)

#     return_layers = {'3': '0', '6': '1', '13': '2', '18': '3'}
#     if backbone_name=='mobilenet_v2_50':
#         in_channels_list = [16,16,48,1280]
#     elif backbone_name=='mobilenet_v2_75':
#         in_channels_list = [24,24,72,1280]
#     elif backbone_name=='mobilenet_v2_100':
#         in_channels_list = [32,32,96,1280]
#     return BackboneWithFPN(backbone, return_layers, in_channels_list, fpn_out_channels, pyramid_levels)


