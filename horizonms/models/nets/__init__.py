from .vgg import VGG, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg13_bn_narrow, \
                 vgg16, vgg16_bn, vgg19, vgg19_bn
from .resnet import ResNet, resnet18, resnet34, resnet50, resnet101, \
                    resnet152, resnext50_32x4d, resnext101_32x8d, \
                    wide_resnet50_2, wide_resnet101_2
from .densenet import DenseNet, densenet121, densenet169, densenet201, densenet161
from .efficientnet import EfficientNet, efficientnet_b0, efficientnet_b1, \
                          efficientnet_b2, efficientnet_b3, efficientnet_b4, \
                          efficientnet_b4, efficientnet_b5, efficientnet_b6, \
                          efficientnet_b7, efficientnet_b8, efficientnet_l2
from .inception_resnet import InceptionResnetV2, inception_resnet_v2, ens_adv_inception_resnet_v2
from .shufflenetv2 import ShuffleNetV2, shufflenet_v2_x0_5, shufflenet_v2_x1_0, \
                          shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .mobilenetv2 import MobileNetV2, mobilenet_v2
from .resenetrs import ResnetRS, resnetrs50, resnetrs101, resnetrs152, resnetrs200
from .nfnet import NFNet, nfnet_f0, nfnet_f1, nfnet_f2, nfnet_f3, nfnet_f4, nfnet_f5
from .vgg_like import VGGLike, vgg_like_v1, vgg_like_v2, vgg_like_v3
from .vgg_se import VGG_SE


__all__ = ["VGG", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg13_bn_narrow", 
           "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
           "ResNet", "resnet18", "resnet34", "resnet50", "resnet101",
           "resnet152", "resnext50_32x4d", "resnext101_32x8d",
           "wide_resnet50_2", "wide_resnet101_2",
           "DenseNet", "densenet121", "densenet169", "densenet201", "densenet161",
           "EfficientNet", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
           "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6",
           "efficientnet_b7", "efficientnet_b8", "efficientnet_l2",
           "InceptionResnetV2", "inception_resnet_v2", "ens_adv_inception_resnet_v2",
           "ShuffleNetV2", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
           "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
           "MobileNetV2", "mobilenet_v2",
           "ResnetRS", "resnetrs50", "resnetrs101", "resnetrs152", "resnetrs200",
           "EfficientNetMultiHeads", "efficientnet_multiheads_b0", "efficientnet_multiheads_b1",
           "efficientnet_multiheads_b2", "efficientnet_multiheads_b3", "efficientnet_multiheads_b4",
           "efficientnet_multiheads_b5", "efficientnet_multiheads_b6", "efficientnet_multiheads_b7",
           "efficientnet_multiheads_b8", "efficientnet_multiheads_l2",
           "EfficientNetMixMultiHeads", "efficientnet_mixmultiheads_b0", "efficientnet_mixmultiheads_b1",
           "efficientnet_mixmultiheads_b2", "efficientnet_mixmultiheads_b3", "efficientnet_mixmultiheads_b4",
           "efficientnet_mixmultiheads_b5", "efficientnet_mixmultiheads_b6", "efficientnet_mixmultiheads_b7",
           "efficientnet_mixmultiheads_b8", "efficientnet_mixmultiheads_l2",
           "NFNet", "nfnet_f0", "nfnet_f1", "nfnet_f2", "nfnet_f3", "nfnet_f4", "nfnet_f5",
           "NFNetMultiHeads", "nfnet_multiheads_f0", "nfnet_multiheads_f1", "nfnet_multiheads_f2",
           "nfnet_multiheads_f3", "nfnet_multiheads_f4", "nfnet_multiheads_f5",
           "NFNetMixMultiHeads", "nfnet_mixmultiheads_f0", "nfnet_mixmultiheads_f1", "nfnet_mixmultiheads_f2",
           "nfnet_mixmultiheads_f3", "nfnet_mixmultiheads_f4", "nfnet_mixmultiheads_f5",
           "VGGLike", "vgg_like_v1", "vgg_like_v2", "vgg_like_v3",
           "VGG_SE"
]


# def get_net(net_params):
#     net_name = net_params["net_name"]
#     net_params.pop("net_name")
#     net = eval(net_name)(**net_params)
#     return net
