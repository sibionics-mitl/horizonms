import torch
from torch import nn
from ..ops.misc import SqueezeExcitation, ConvNormActivation
from ..ops.stochastic_depth import StochasticDepth
import numbers
from ...builder import NETS


__all__ = ("VGGLike", "vgg_like_v1", "vgg_like_v2", "vgg_like_v3")


class ConvNormActivationDrop(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=3, padding=1,
                 stochastic_depth_prob=0):
        super(ConvNormActivationDrop, self).__init__()
        self.stochastic_depth_prob = stochastic_depth_prob
        self.conv_bn = ConvNormActivation(in_channels, out_channles, 
                                kernel_size=kernel_size, padding=padding)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.conv_bn(input)
        if self.stochastic_depth_prob > 0:
            result = self.stochastic_depth(result)
            result += input
        return result


class Block(nn.Module):
    def __init__(self, input_dim, cfg, ratio=16, kernel_size=3, padding=1,
                 stochastic_depth_prob=0.2):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.ratio = ratio
        self.kernel_size= kernel_size
        self.padding = padding
        self.stochastic_depth_prob = stochastic_depth_prob
        self.block = self.make_layers(cfg)

    def make_layers(self, cfg):
        layers = []
        in_channels = self.input_dim
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'se':
                layers += [SqueezeExcitation(in_channels, in_channels//self.ratio)]
            else:
                layers += [ConvNormActivationDrop(in_channels, v, kernel_size=self.kernel_size,
                        padding=self.padding, stochastic_depth_prob=self.stochastic_depth_prob)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, input):
        result = self.block(input)
        # if self.use_res_connect:
        #     result = self.stochastic_depth(result)
        #     result += input
        return result


@NETS.register_module()
class VGGLike(nn.Module):
    def __init__(self, input_dim, num_classes=1000, init_weights=True, cfgs=None, dropout=0.5):
        super(VGGLike, self).__init__()
        self.input_dim = input_dim

        v_dim = input_dim
        self.blocks = nn.ModuleList()
        def get_outchannels(cfg):
            cfg = cfg[::-1]
            for k in range(len(cfg)):
                out = cfg[k]
                if isinstance(out, numbers.Number):
                    return out

        for cfg in cfgs:
            self.blocks.append(Block(v_dim, **cfg))
            v_dim = get_outchannels(cfg['cfg'])

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(v_dim * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

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
    
    def forward(self, x):
        print(x.shape)
        for block in self.blocks:
            x = block(x)
            print(x.shape)
        x = torch.cat([x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))], dim=1)
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)
        return x


@NETS.register_module()
def vgg_like_v1(input_dim, num_classes, channels=16, stochastic_depth_prob=0.2, dropout=0.5):
    comm = dict(ratio=16, stochastic_depth_prob=stochastic_depth_prob)
    cfgs = [dict(cfg=[channels, channels, channels, 'se', 'M']).update(comm),
            dict(cfg=[2*channels, 2*channels, 2*channels, 'se', 'M']).update(comm),
            dict(cfg=[4*channels, 4*channels, 4*channels, 'se', 'M']).update(comm),
            dict(cfg=[8*channels, 8*channels, 8*channels, 8*channels, 'se', 'M']).update(comm),
            dict(cfg=[16*channels, 16*channels, 16*channels, 16*channels, 'se', 'M']).update(comm)]
    return VGGLike(input_dim, num_classes, True, cfgs, dropout)


@NETS.register_module()         
def vgg_like_v2(input_dim, num_classes, channels=16, stochastic_depth_prob=0.2, dropout=0.5):
    comm = dict(ratio=16, stochastic_depth_prob=stochastic_depth_prob)
    cfgs = [dict(cfg=[channels, channels, channels, 'se', 'M']).update(comm),
            dict(cfg=[2*channels, 2*channels, 2*channels, 'se', 'M']).update(comm),
            dict(cfg=[4*channels, 4*channels, 4*channels, 'se', 'M']).update(comm),
            dict(cfg=[8*channels, 8*channels, 8*channels, 8*channels, 'se', 'M']).update(comm),
            dict(cfg=[16*channels, 16*channels, 16*channels, 16*channels, 'se', 'M']).update(comm),
            dict(cfg=[32*channels, 32*channels, 32*channels, 32*channels, 'se', 'M']).update(comm)]
    return VGGLike(input_dim, num_classes, True, cfgs, dropout)


@NETS.register_module()
def vgg_like_v3(input_dim, num_classes, channels=16, stochastic_depth_prob=0.2, dropout=0.5):
    comm = dict(ratio=16, stochastic_depth_prob=stochastic_depth_prob)
    cfgs = [dict(cfg=[channels, channels, channels, 'se', 'M']).update(comm),
            dict(cfg=[2*channels, 2*channels, 2*channels, 'se', 'M']).update(comm),
            dict(cfg=[4*channels, 4*channels, 4*channels, 'se', 'M']).update(comm),
            dict(cfg=[8*channels, 8*channels, 8*channels, 8*channels, 'se', 'M']).update(comm),
            dict(cfg=[16*channels, 16*channels, 16*channels, 16*channels, 'se', 'M']).update(comm),
            dict(cfg=[32*channels, 32*channels, 32*channels, 32*channels, 'se', 'M']).update(comm),
            dict(cfg=[64*channels, 64*channels, 64*channels, 64*channels, 'se', 'M']).update(comm)]
    return VGGLike(input_dim, num_classes, True, cfgs, dropout)