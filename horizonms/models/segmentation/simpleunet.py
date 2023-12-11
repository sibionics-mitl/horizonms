import math
import torch.nn as nn
from .decoder import UnetSimpleDecoder
from ...builder import NETS


__all__ = ("SimpleUNet")


class SimpleEnconderModule(nn.Module):
    def __init__(self, cfg, in_channels, pooling=True):
        super().__init__()
        self.cfg         = cfg
        self.in_channels = in_channels
        self.pooling = pooling
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = self.make_layers()
        
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

    def make_layers(self, batch_norm=True):
        in_channels = self.in_channels
        layers = []
        for v in self.cfg:
            conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            conv2 = nn.Conv2d(in_channels, v, kernel_size=1, padding=0)
            if batch_norm:
                layers += [conv1, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                           conv2, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1, nn.ReLU(inplace=True), conv2, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.pooling:
            x = self.down(x)
        x = self.conv(x)
        return x


@NETS.register_module()
class SimpleUNet(nn.Module):
    r"""A simple Unet. Its encoder and decoder are composed of simple cascade of Conv, BN and ReLu layers.

    Args:
        input_dim (int): the dimension of input.
        num_classes (int): the number of classes for segmentation.
        num_block (int): the number of blocks, each block is associated with a downsampling in encoder.
        channels_in (int): the number of channels in the first Conv.
        prior (float): the parameter used to estimate the initilization of the bias of the last Conv.
    """
    def __init__(self, input_dim: int, num_classes: int, num_block: int = 5,
                 channels_in: int = 24, prior: float = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_block = num_block
        self.channels_in = channels_in

        # Encoder
        channels = self.input_dim
        self.down_list = nn.ModuleList()
        for k in range(self.num_block):
            if k >= 3:
                out_channels = (2**3) * self.channels_in
            else:
                out_channels = (2**k) * self.channels_in
            if k == 0:
                self.down_list.append(SimpleEnconderModule([out_channels, out_channels],
                        channels, pooling=False))
            else:
                self.down_list.append(SimpleEnconderModule([out_channels, out_channels],
                        channels, pooling=True))
            channels = out_channels

        # Decoder
        factor_in = 2**min(3, self.num_block-1) + 2**min(3, self.num_block-2)
        self.up_list = nn.ModuleList()
        for k in range(self.num_block-1, 0, -1):
            if k == 0:
                factor_out = 1
            else:
                factor_out = 2**(k-1)               
            self.up_list.append(UnetSimpleDecoder([factor_out*self.channels_in, 
                                    factor_out*self.channels_in], 
                                    factor_in*self.channels_in))
            factor_in = factor_out + 2**(k-2)
            
        self.out = nn.Conv2d(self.channels_in, self.num_classes, kernel_size=1)

        self.init_weights()
        print(f"Initialized {self.__class__.__name__} succesfully")

        # initialization of segmentation output
        if prior is not None:
            bias_value = -(math.log((1 - prior) / prior))
            self.out.bias.data.fill_(bias_value)

    def forward(self, x):
        features = []
        for module in self.down_list:
            x = module(x)
            features.append(x)

        x = features[-1]
        for k, module in enumerate(self.up_list):
            x = module(x, features[-k-2])

        x = self.out(x)

        return x

    def init_weights(self):
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
