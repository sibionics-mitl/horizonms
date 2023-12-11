import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from ...builder import NETS


__all__ = ("VGG", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg13_bn_narrow", 
           "vgg16", "vgg16_bn", "vgg19", "vgg19_bn")


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


@NETS.register_module()
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
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


def make_layers(cfg, input_dim, batch_norm=False):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, input_dim, cfg, batch_norm, pretrained, progress, model_dir='.', **kwargs):
    if len(pretrained) > 0:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], input_dim, batch_norm=batch_norm), **kwargs)
    if (pretrained == 'imagenet') & (input_dim == 3):
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              model_dir=model_dir,
                                              progress=progress)
        model.load_state_dict(state_dict)
    elif len(pretrained) > 0:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model


@NETS.register_module()
def vgg11(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg11', input_dim, 'A', False, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg11_bn(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg11_bn', input_dim, 'A', True, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg13(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg13', input_dim, 'B', False, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg13_bn(input_dim, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg13_bn', input_dim, 'B', True, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg13_bn_narrow(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg13_bn_narrow', input_dim, 'C', True, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg16(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg16', input_dim, 'D', False, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg16_bn(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg16_bn', input_dim, 'D', True, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg19(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg19', input_dim, 'E', False, pretrained, progress, model_dir=model_dir, **kwargs)


@NETS.register_module()
def vgg19_bn(input_dim=3, pretrained='', progress=True, model_dir='.', **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return _vgg('vgg19_bn', input_dim, 'E', True, pretrained, progress, model_dir=model_dir, **kwargs)
