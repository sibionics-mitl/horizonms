from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict


__all__ = ("PseudomaxPooling2D", "ConvNormActivation", "SqueezeExcitation")


class PseudomaxPooling2D(torch.nn.Module):
    def __init__(self, mode='alpha-softmax', alpha=4.0, kernel_size=3, stride=1, padding=0):
        super().__init__()
        assert mode in ['alpha-softmax', 'alpha-quasimax'], \
                    "alpha has to be 'alpha-softmax' or 'alpha-quasimax'"
        self.mode = mode
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_padding = (self.kernel_size - 1) // 2
        assert padding <= self.conv_padding, \
                "padding has to be less than or equal to (kernel_size - 1) // 2"
        self.start = self.conv_padding - self.padding
        self.epsilon = 1e-8
        # print("padding", padding, self.start)

    def forward(self, x):
        c = x.shape[1]
        weight = torch.ones(c, 1, self.kernel_size, self.kernel_size, device=x.device)
        if self.mode == 'alpha-softmax':
            numerator = F.conv2d(x*torch.exp(self.alpha*x), weight, padding=self.conv_padding, groups=c)
            denominator = F.conv2d(torch.exp(self.alpha*x), weight, padding=self.conv_padding, groups=c)
            pmax = numerator / (denominator + self.epsilon)
            # print("input", x[0, 0])
            # print("pmax", pmax[0, 0])
        else:
            pmax = F.conv2d(torch.exp(self.alpha*x), weight, padding=self.conv_padding, groups=c)
            pmax = (torch.log(pmax) - 2*math.log(float(self.kernel_size)))/self.alpha
        return pmax[:, :, self.start::self.stride, self.start::self.stride]

    def __repr__(self):
        s = '{name}('
        d = dict(self.__dict__)
        keys = ['mode', 'alpha', 'kernel_size', 'stride', 'padding', 'conv_padding']
        d = {key: value for key, value in d.items() if key in keys}
        for key, value in d.items():
                s += f'{key}={value}, '
        s = s[:-2] + ')'
        return s.format(name=self.__class__.__name__, **d)


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        named: bool = False,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation=dilation, groups=groups, bias=norm_layer is None,)
        layers = [conv]
        names = ['conv']
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            names.append('normalization')
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
            names.append('activation')
        if named:
            layers = OrderedDict([(name, layer) for name, layer in zip(names, layers)])
            super().__init__(layers)
        else:
            super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input