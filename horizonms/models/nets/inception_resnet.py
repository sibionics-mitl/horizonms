""" 
Pytorch Inception-Resnet-V2 implementation
Sourced from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_resnet_v2.py
Sourced from https://github.com/Cadene/tensorflow-model-zoo.torch (MIT License) which is
based upon Google's Tensorflow implementation and pretrained weights (Apache 2.0 License)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from .utils import load_state_dict_from_url
from ...builder import NETS


__all__ = ["InceptionResnetV2", "inception_resnet_v2", "ens_adv_inception_resnet_v2"]


model_urls = {
    "inception_resnet_v2": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/inception_resnet_v2-940b1cd6.pth",
    "ens_adv_inception_resnet_v2": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ens_adv_inception_resnet_v2-2592a550.pth",
}


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False):
        super(Block8, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out


@NETS.register_module()
class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=1000, input_dim=3, drop_rate=0.):
        super(InceptionResnetV2, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536

        conv2d_1a = BasicConv2d(input_dim, 32, kernel_size=3, stride=2)
        conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.feature_info = [dict(num_chs=64, reduction=2, module='conv2d_2b')]

        maxpool_3a = nn.MaxPool2d(3, stride=2)
        conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.feature_info += [dict(num_chs=192, reduction=4, module='conv2d_4a')]

        maxpool_5a = nn.MaxPool2d(3, stride=2)
        mixed_5b = Mixed_5b()
        repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]

        mixed_6a = Mixed_6a()
        repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]

        mixed_7a = Mixed_7a()
        repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        block8 = Block8(no_relu=True)
        conv2d_7b = BasicConv2d(2080, self.num_features, kernel_size=1, stride=1)
        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv2d_7b')]

        # self.global_pool, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.features = nn.Sequential(conv2d_1a, conv2d_2a, conv2d_2b,
                    maxpool_3a, conv2d_3b, conv2d_4a, maxpool_5a, mixed_5b, repeat,
                    mixed_6a, repeat_1, mixed_7a, repeat_2, block8, conv2d_7b)

        # self.global_pool = Pool2d(flatten=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)#nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.classifier = nn.Linear(self.num_features, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


def create_inception_resnet_v2(arch, input_dim: int = 3, pretrained: str = 'imagenet',
        progress: bool = True, model_dir: str = '.', **kwargs: Any) -> InceptionResnetV2:
    model = InceptionResnetV2(input_dim=input_dim, **kwargs)
    if (pretrained == 'imagenet') & (input_dim == 3):
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], model_dir=model_dir, progress=progress)
        model.load_state_dict(state_dict)
    elif len(pretrained) > 0:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model


@NETS.register_module()
def inception_resnet_v2(input_dim: int = 3, pretrained: str = '',
        progress: bool = True, model_dir: str = '.', **kwargs: Any) -> InceptionResnetV2:
    r"""InceptionResnetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>` paper.

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return create_inception_resnet_v2("inception_resnet_v2", input_dim, pretrained,
                progress, model_dir, **kwargs)


@NETS.register_module()
def ens_adv_inception_resnet_v2(input_dim: int = 3, pretrained: str = '',
        progress: bool = True, model_dir: str = '.', **kwargs: Any) -> InceptionResnetV2:
    r""" Ensemble Adversarially trained InceptionResnetV2 model architecture
    As per https://arxiv.org/abs/1705.07204 and
    https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models.

    Args:
        input_dim (int): dimension of input.
        pretrained (str): specifies directory of pretrained model. `pretrained='imagenet'` uses ImageNet pretrained model.
        progress (bool): if True, displays a progress bar of the download to stderr.
        model_dir (str): directory of pretrained model.
    """
    return create_inception_resnet_v2("ens_adv_inception_resnet_v2", input_dim, pretrained,
                progress, model_dir, **kwargs)