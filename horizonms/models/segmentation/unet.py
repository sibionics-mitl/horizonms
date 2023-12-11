import torch
from torch import nn
import math
import numpy as np
from .decoder import UnetSimpleDecoder
from ...builder import NETS, build_backbone


__all__ = ("UNet_V1", "UNet_V2")


@NETS.register_module()
class UNet_V1(nn.Module):
    r"""A simple Unet. Its encoder is obtained from backbone, and its decoder is a simple cascade of Conv, BN and ReLu layers.

    Args:
        backbone (Dict): the configuration of the backbone.
        cfg_up (Dict): the configuration of the decoder, which is  a simple cascade of Conv, BN and ReLu layers.
        final_activation ('softmax' | 'sigmoid' | None): Decide which type of operator is used to the output of `net`.
            When final_activation=None, no operator is applied.
            When final_activation='softmax', softmax operator is applied.
            When final_activation='softmax', sigmoid operator is applied.
        cfg_seg (Dict): the configuration of the head in decoder.
        nb_features (int): the number of the levels of the features.
        nb_output (int): the number of the multiple resolution outputs.
        num_classes (int): the number of classes for segmentation.
        prior (float): the parameter used to estimate the initilization of the bias of the last Conv.
    """
    def __init__(self, backbone, cfg_up, final_activation=None,
                 cfg_seg=[128,64], nb_features=1, nb_output=1,
                 num_classes=2, prior=None):
        super(UNet_V1, self).__init__()
        assert final_activation in ['softmax', 'sigmoid', None]
        self.encoder = build_backbone(backbone)
        in_channels_list = self.encoder.out_channels
        nb_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print('# trainable parameters in backbone: {}'.format(nb_params))
        
        self.nb_features = nb_features
        self.nb_output = nb_output
        self.num_classes = num_classes
        self.stages = len(in_channels_list)
        assert self.nb_output<=self.nb_features
        assert len(cfg_up) == len(in_channels_list) - 1

        cfg_up_invert = cfg_up[::-1]
        up_channels = [cfg[-1] for cfg in cfg_up] + [in_channels_list[-1]]
        self.seg_feats = nn.ModuleList()
        self.outs = nn.ModuleList()
        for k in range(self.nb_features):
            seg_feat = self.make_layers(cfg_seg, in_channels=up_channels[k])
            if final_activation == 'softmax':
                out = nn.Sequential(
                    nn.Conv2d(cfg_seg[-1], num_classes, kernel_size=1),
                    nn.Softmax(dim=1)
                    )
            elif final_activation == 'sigmoid':
                out = nn.Sequential(
                    nn.Conv2d(cfg_seg[-1], num_classes, kernel_size=1),
                    nn.Sigmoid()
                    )
            else:
                out = nn.Conv2d(cfg_seg[-1], num_classes, kernel_size=1)
            self.seg_feats.append(seg_feat)
            self.outs.append(out)
            
        nb_params = []
        for model in self.seg_feats:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in segmentation features: {}'.format(np.sum(nb_params)))

        nb_params = []
        for model in self.outs:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in segmentation outputs: {}'.format(np.sum(nb_params)))

        self._initialize_weights()

        # initialization of segmentation output
        if prior is not None:
            bias_value = -(math.log((1 - prior) / prior))
            for modules in self.outs:
                for layer in modules.modules():
                    if isinstance(layer, nn.Conv2d):
                        layer.bias.data.fill_(bias_value)    

        self.decoder = nn.ModuleList()
        up_channels1 = [in_channels_list[-1]]+[cfg[-1] for cfg in cfg_up][::-1][:self.stages-2]
        up_channels2 = in_channels_list[:self.stages-1][::-1]
        up_channels = [up1+up2 for up1, up2 in zip(up_channels1, up_channels2)]
        print("**",up_channels1)
        print("**",up_channels2)
        print("**up channels: ", up_channels)
        
        for k in range(self.stages-1):
            up = UnetSimpleDecoder(cfg_up_invert[k], up_channels[k])
            self.decoder.append(up)
        nb_params = []
        for model in self.decoder:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in decoder network: {}'.format(np.sum(nb_params)))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, in_channels, batch_norm=True):
        layers = []
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

    def forward(self, images):
        x = self.encoder(images)
        x = list(x)[::-1]
        y = [x[0]]
        for k, up_block in enumerate(self.decoder):
            y.append(up_block(y[-1], x[k+1]))

        y = y[::-1]
        seg_preds = []
        for k, (seg_feat, out) in enumerate(zip(self.seg_feats, self.outs)):
            seg_preds.append(out(seg_feat(y[k])))
        assert len(seg_preds)==self.nb_features
        # print("<<< predictiutilsons: ", [v.shape for v in seg_preds])
        if self.nb_output == 1:
            return seg_preds[0]
        else:
            return tuple(seg_preds[:self.nb_output])


@NETS.register_module()
class UNet_V2(UNet_V1):
    r"""A simple Unet. Its encoder is obtained from backbone, and its decoder is a simple cascade of Conv, BN and ReLu layers.

    Args:
        backbone (Dict): the configuration of the backbone.
        cfg_up (Dict): the configuration of the decoder, which is  a simple cascade of Conv, BN and ReLu layers.
        final_activation ('softmax' | 'sigmoid' | None): Decide which type of operator is used to the output of `net`.
            When final_activation=None, no operator is applied.
            When final_activation='softmax', softmax operator is applied.
            When final_activation='softmax', sigmoid operator is applied.
        cfg_seg (Dict): the configuration of the head in decoder.
        nb_features (int): the number of the levels of the features.
        nb_output (int): the number of the multiple resolution outputs.
        num_classes (int): the number of classes for segmentation.
        prior (float): the parameter used to estimate the initilization of the bias of the last Conv.
    """
    def __init__(self, backbone, 
                 cfg_up, cfg_seg=[128,64], nb_features=1, nb_output=1, num_classes=2, 
                 prior=None):
        super(UNet_V2, self).__init__(backbone, cfg_up, cfg_seg, 
                    nb_features, nb_output, num_classes, prior)

        self.outs = nn.ModuleList()
        for k in range(self.nb_features):
            out_feat = cfg_seg[-1]+num_classes
            if k==self.nb_features-1:
                out_feat = cfg_seg[-1]
            out = nn.Conv2d(out_feat, num_classes, kernel_size=1)
            self.outs.append(out)
        self.outs = self.outs[::-1]

        self.up_pred = self.pred_upsampling(2)

        if prior is not None:
            bias_value = -(math.log((1 - prior) / prior))
        for modules in self.outs:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None and prior is not None:
                        nn.init.constant_(layer.bias, bias_value)
        
    def pred_upsampling(self, scale_factor):
        layers = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return layers

    def forward(self, images):
        x = self.encoder(images)
        x = list(x)[::-1]
        y = [x[0]]
        for k, up_block in enumerate(self.decoder):
            y.append(up_block(y[-1], x[k+1]))

        y = y[::-1]
        z = []
        for k, seg_feat in enumerate(self.seg_feats):
            z.insert(0, seg_feat(y[k]))

        seg_preds = []
        for k, out in enumerate(self.outs):
            if k==0:
                logit = out(z[k])
            else:
                logit = out(torch.cat([z[k], self.up_pred(seg_preds[-1])], dim=1))
            seg_preds.append(logit)
        seg_preds = seg_preds[::-1]    
        assert len(seg_preds)==self.nb_features
        if self.nb_output == 1:
            return seg_preds[0]
        else:
            return tuple(seg_preds[:self.nb_output])