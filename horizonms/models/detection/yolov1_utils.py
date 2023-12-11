import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...builder import BACKBONES, NECKS, HEADS


__all__ = ("DefaultDarknet", "DefaultYolov1Head", "DetNeckBlock", "BottlenetNeck", "Yolov1Head")


class LeakyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LeakyConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrulu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrulu(self.bn(self.conv(x)))


"""
Each conv layer is a tuple (kernel_size, out_ch, stride, padding).
Each conv block is a list [(conv1_params), ... , (convN_params), num_repeats].
"M" --> MaxPool2d with stride 2 and size 2.
"""
YOLOV1_CFG = [
            (7, 64, 2, 3), "M",
            (3, 192, 1, 1), "M",
            (1, 128, 1, 0),
            (3, 256, 1, 1),
            (1, 256, 1, 0),
            (3, 512, 1, 1), "M",
            [(1, 256, 1, 0), (3, 512, 1, 1), 4],
            (1, 512, 1, 0),
            (3, 1024, 1, 1), "M",
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1024, 1, 1),
            (3, 1024, 2, 1),
            (3, 1024, 1, 1),
            (3, 1024, 1, 1)]


@BACKBONES.register_module()
class DefaultDarknet(nn.Module):
    def __init__(self, in_channels, cfg=YOLOV1_CFG):
        super(DefaultDarknet, self).__init__()
        self.in_channels = in_channels
        self.cfg = cfg
        layers = []
        in_ch = self.in_channels
        for x in self.cfg:
            if type(x) == tuple:
                # * convolution
                layers += [LeakyConvBlock(in_channels=in_ch, out_channels=x[1],
                    kernel_size=x[0], stride=x[2], padding=x[3])]
                in_ch = x[1]
            elif type(x) == str:
                # * add max pooling layer
                layers += [nn.MaxPool2d(2,2)]
            elif type(x) == list:
                # * ConvBlock
                convs, num_repeat = x[:-1], x[-1]                
                for _ in range(num_repeat):
                    for conv in convs:
                        layers += [LeakyConvBlock(in_channels=in_ch, out_channels=conv[1],
                            kernel_size=conv[0], stride=conv[2], padding=conv[3])]
                        in_ch = conv[1]
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.backbone(x)


@HEADS.register_module()
class DefaultYolov1Head(nn.Module):
    def __init__(self, feature_shape, num_classes, num_boxes):
        super(DefaultYolov1Head, self).__init__()
        self.feature_shape = feature_shape
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024*feature_shape[0]*feature_shape[1], 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, feature_shape[0]*feature_shape[1]*(num_classes+num_boxes*5)),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.head(x)
        x = x.reshape(x.shape[0], -1, self.feature_shape[0], self.feature_shape[1])
        # classification, [B, num_classes, H, W]
        pred_cls = x[:, :self.num_classes]
        # confidence, [B, num_boxes, H, W]
        pred_confidence = x[:, self.num_classes:self.num_classes+self.num_boxes]
        # bboxes, [B, 4*num_boxes, H, W]
        pred_txtytwth = x[:, self.num_classes+self.num_boxes:]
        return pred_cls, pred_confidence, pred_txtytwth


class DetNeckBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, block_type='A', expansion=1):
        super(DetNeckBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


@NECKS.register_module()
class BottlenetNeck(nn.Module):
    def __init__(self, in_channels, out_channels=256, stride=1, expansion=1):
        super(BottlenetNeck, self).__init__()
        self.neck_block0 = DetNeckBlock(in_channels, out_channels, stride, 'B', expansion)
        self.neck_block1 = DetNeckBlock(out_channels, out_channels, stride, 'A', expansion)
        self.neck_block2 = DetNeckBlock(out_channels, out_channels, stride, 'A', expansion)

    def forward(self, x):
        x = self.neck_block0(x)
        x = self.neck_block1(x)
        x = self.neck_block2(x)
        return x


@HEADS.register_module()
class Yolov1Head(nn.Module):
    def __init__(self, in_channels, num_classes, num_boxes, prior=0.02):
        super(Yolov1Head, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.conv = nn.Conv2d(in_channels, num_classes+num_boxes*5, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_classes+num_boxes*5)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.sigmoid(self.bn(self.conv(x)))
        # classification, [B, num_classes, H, W]
        pred_cls = x[:, :self.num_classes]
        # confidence, [B, num_boxes, H, W]
        pred_confidence = x[:, self.num_classes:self.num_classes+self.num_boxes]
        # bboxes, [B, 4*num_boxes, H, W]
        pred_txtytwth = x[:, self.num_classes+self.num_boxes:]
        return pred_cls, pred_confidence, pred_txtytwth
        # return x