from torch import nn
from torchvision.ops import misc as misc_nn_ops
from .base import Backbone
from ..nets import resnet, vgg, densenet, mobilenetv2
from ...builder import BACKBONES


__all__ = ("vgg_unet_backbone", "resnet_unet_backbone", 
           "densenet_unet_backbone", "mobilenetv2_unet_backbone")


@BACKBONES.register_module()
def vgg_unet_backbone(backbone_name, input_dim=3, pretrained=False, model_dir='.', trainable_layers=10):
    r"""It extracts a VGG backbone for UNet.
    
    Args:
        backbone_name (str): the name of backbone.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_layers (int): the number of trainable (not frozen) layers starting from the last layer.
    """   
    backbone = vgg.__dict__[backbone_name](input_dim, pretrained=pretrained, model_dir=model_dir)
    backbone = backbone.features

    # select layers that wont be frozen
    assert trainable_layers >= 0
    layers_to_train = []
    pool_layers = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            layers_to_train.insert(0,name)
        elif isinstance(module, nn.MaxPool2d):
            pool_layers.append(name)
    if trainable_layers<len(layers_to_train):
        layers_to_train = layers_to_train[:trainable_layers]

    cand_return_layers = [str(int(f)-1)for f in pool_layers]
    
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {cand: str(k) for k,cand in enumerate(cand_return_layers)}
    if 'narrow' in backbone_name:
        in_channels_list = [32,64,128,256,256]
    else:
        in_channels_list = [64,128,256,512,512]
    return Backbone(backbone, return_layers, in_channels_list)


@BACKBONES.register_module()
def resnet_unet_backbone(backbone_name, input_dim=3, pretrained=False, model_dir='.',
                    norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    r"""It extracts a ResNet backbone for UNet.
    
    Args:
        backbone_name (str): the name of backbone.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_layers (int): the number of trainable (not frozen) stages starting from the last stage.
    """   
    backbone = resnet.__dict__[backbone_name](input_dim, pretrained=pretrained, 
                                              model_dir=model_dir, norm_layer=norm_layer)
    # print(backbone)

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'relu':'0','layer1':'1','layer2':'2','layer3':'3','layer4':'4'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [64,in_channels_stage2,in_channels_stage2*2,
                        in_channels_stage2*4, in_channels_stage2*8]
    return Backbone(backbone, return_layers, in_channels_list)


@BACKBONES.register_module()
def densenet_unet_backbone(backbone_name, input_dim=3, pretrained=False, model_dir='.', trainable_layers=3):
    r"""It extracts a DenseNet backbone for UNet.
    
    Args:
        backbone_name (str): the name of backbone.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_layers (int): the number of trainable (not frozen) stages starting from the last stage.
    """  
    backbone = densenet.__dict__[backbone_name](input_dim, pretrained=pretrained, model_dir=model_dir)
    backbone = backbone.features

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['denseblock4', 'denseblock3', 'denseblock2', 'denseblock1', 'conv0'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'relu0':'0','denseblock1':'1', 'denseblock2':'2', 'denseblock3':'3', 'denseblock4':'4'}
    if backbone_name=='densenet121':
        in_channels_list = [64,256,512,1024,1024]
    elif backbone_name=='densenet169':
        in_channels_list = [64,256,512,1280,1664]
    elif backbone_name=='densenet201':
        in_channels_list = [64,256,512,1792,1920]
    elif backbone_name=='densenet161':
        in_channels_list = [96,384,768,2112,2208]
    return Backbone(backbone, return_layers, in_channels_list)


@BACKBONES.register_module()
def mobilenetv2_unet_backbone(backbone_name, input_dim=3, pretrained=False, model_dir='.', trainable_layers=3):
    r"""It extracts a MobileNetv2 backbone for UNet.
    
    Args:
        backbone_name (str): the name of backbone.
        input_dim (int): the dimension of input.
        pretrained (bool): whether to use pretrained weights when extracting.
        model_dir (str): the directory to save the pretrained weights.
        trainable_layers (int): the number of trainable (not frozen) stages starting from the last stage.
    """  
    in_channels_list = [16,24,32,96,320]
    if backbone_name=='mobilenetv2_1.0':
        width_mult=1.0
    elif backbone_name=='mobilenetv2_0.75':
        width_mult=0.75
    elif backbone_name=='mobilenetv2_0.5':
        width_mult=0.5
    elif backbone_name=='mobilenetv2_0.25':
        width_mult=0.25
    backbone = mobilenetv2.mobilenet_v2(input_dim=input_dim, width_mult=width_mult,
                                            pretrained=pretrained, model_dir=model_dir)
    backbone = backbone.features

    # select layers that wont be frozen
    assert trainable_layers >= 0
    layers_to_train = ['denseblock4', 'denseblock3', 'denseblock2', 'denseblock1', 'conv0'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'1':'0','3':'1', '6':'2', '13':'3', '17':'4'}
    in_channels_list = [int(v*width_mult) for v in in_channels_list]
    return Backbone(backbone, return_layers, in_channels_list)
