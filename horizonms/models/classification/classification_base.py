import torch
from torch import nn
from abc import ABC, abstractmethod
from ..model_base import BaseModel
from ...builder import build_net, build_backbone, build_neck, build_head, MODELS, NETS


__all__ = ("BaseClassification", "ClassificationNetFromModules", "get_classification_net")


@MODELS.register_module()
class BaseClassification(BaseModel, ABC):
    r"""Base class for classification task.

    Args:
        net (nn.Module): Deep learning network.
        final_activation ('softmax' | 'sigmoid' | None): Decide which type of operator is used to the output of `net`.
            When final_activation=None, no operator is applied.
            When final_activation='softmax', softmax operator is applied.
            When final_activation='softmax', sigmoid operator is applied.
        batch_image: class used to convert a list of (input, target) into batch format used in network training and testing.
        divisible: it determines the size of the batched input such that it is divisible by `divisible` and larger than the size of the input.
        batch_transforms: batch transformation for network training.
    """
    def __init__(self, net, final_activation, batch_image, divisible=1, batch_transforms=None):
        super(BaseClassification, self).__init__(net, final_activation)

        # self.net = net

        self.batch_image = None
        if batch_image is not None:
            self.batch_image = batch_image(divisible)

        self.batch_transforms = batch_transforms

    def preprocessing_input(self, images, targets=None):
        if self.batch_image is not None:
            images, targets = self.batch_image(images, targets)
        if self.batch_transforms is not None:
            images, targets = self.batch_transforms(images, targets)
        return images, targets

    @abstractmethod
    def calculate_losses(self, targets, ypred):
        pass 

    @abstractmethod
    def calculate_metrics(self, targets, ypred):
        pass

    def forward_train(self, images, targets):
        images, targets = self.preprocessing_input(images, targets)
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')   
        ypred = self.forward(images)

        losses = dict()
        if targets is None:
            return losses, ypred
            
        losses = self.calculate_losses(targets, ypred)
        return losses, ypred

    @torch.no_grad()
    def test_one_batch(self, images, targets):
        images, targets = self.preprocessing_input(images, targets)
        if torch.isnan(images).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images).sum()>0:
            print('image is inf ..............')   
        ypred = self.forward(images)
        losses = self.calculate_losses(targets, ypred)
        metrics = self.calculate_metrics(targets, ypred)
        return losses, metrics, ypred

    @torch.no_grad()
    def predict_one_batch(self, images):
        images, _ = self.preprocessing_input(images, None)
        ypred = self.forward(images)   
        return ypred


@NETS.register_module()
class ClassificationNetFromModules(nn.Module):
    r"""It constructs network by providing 'backbone', 'neck', and 'head'.

    Args:
        cfg (Dict): the configuration of the network.
    """
    def __init__(self, cfg):
        super(ClassificationNetFromModules, self).__init__()
        self.keys = list(cfg.keys())
        assert "backbone" in self.keys, "'backbone' has to be in cfg!"
        self.backbone = build_backbone(cfg["backbone"])
        in_channels = self.backbone.out_channels
        if "neck" in self.keys:
            self.neck = build_neck(cfg["neck"])
            if hasattr(self.neck, "out_channels"):
                in_channels = self.neck.out_channels
        if "head" in self.keys:
            cfg_head = cfg["head"]
            cfg_head.update(dict(input_dim=in_channels))
            self.head = build_head(cfg_head)

    def forward(self, x):
        x = self.backbone(x)
        if "neck" in self.keys:
            x = self.neck(x)
        x = self.head(x)
        return x


def get_classification_net(cfg):
    keys = list(cfg.keys())
    assert ("name" in keys) | ("backbone" in keys), "'name' or 'backbone' has to be provided."
    if "name" in keys:
        net = build_net(cfg)
    else:
        net = ClassificationNetFromModules(cfg)
    return net